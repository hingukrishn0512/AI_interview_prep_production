import os
import json
import time
import logging
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from light_embeddings import LightHFEmbeddings


load_dotenv()

logger = logging.getLogger("interview_coach")
if not logger.handlers:
    # Make sure something actually prints, even if the host app hasn't
    # configured logging itself.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- shared setup (loaded once when this module is imported) ---

search_tool = TavilySearch(max_results=3)
tools = [search_tool]

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
llm_tools = llm.bind_tools(tools)

embedings = LightHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class RateLimitExceeded(Exception):
    """Raised when the LLM provider keeps rate-limiting us after every retry
    has been exhausted, so the API layer can turn this into a clean 503
    instead of a generic 400."""


def safe_llm_invoke(prompt, max_retries: int = 4, base_delay: float = 1.5):
    """Wraps llm.invoke with exponential backoff + jitter.

    This exists because a single DSA request fans out into three Groq calls
    fired back-to-back (arrays/hashing, trees/graphs, dp), which is an easy
    way to trip a provider-side 429 even when overall usage is low. Without
    a retry here, one throttled call takes down the whole /chat request.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            error_text = str(e).lower()
            is_rate_limit = (
                "429" in error_text
                or "rate limit" in error_text
                or "rate_limit" in error_text
                or "too many requests" in error_text
            )
            if not is_rate_limit or attempt == max_retries - 1:
                logger.error("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if is_rate_limit:
                    raise RateLimitExceeded(str(e)) from e
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "LLM call rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, max_retries, delay, e,
            )
            time.sleep(delay)
    # Should be unreachable, but keep mypy/pyflakes happy.
    raise RateLimitExceeded(str(last_error))


def merge_dict(left, right):
    """Custom reducer to merge candidate questions without overwriting."""
    if left is None:
        return right
    if right is None:
        return left
    merged = left.copy()
    merged.update(right)
    return merged


def format_history(state, keep_head: int = 6, keep_tail: int = 20) -> str:
    """Render the conversation history (candidate + coach) as plain text, for
    prompts that need to reason about what's already been said. Without this,
    every node only ever sees the current message in isolation — which is why
    follow-ups, corrections, and "what did I ask earlier" style questions
    previously broke.

    By default the FULL transcript is included, not just a recent window —
    a short window would make "what did I ask at the very beginning?" fail
    again in any conversation longer than a few turns. If the transcript
    gets long, the earliest `keep_head` turns and most recent `keep_tail`
    turns are kept in full (so both "the beginning" and "just now" stay
    answerable) and the middle is collapsed into a one-line note instead of
    being dropped silently.
    """
    past_messages = state.get('messages', []) or []
    turns = []
    for m in past_messages:
        if isinstance(m, tuple):
            speaker, content = m[0], m[1]
        else:
            speaker = getattr(m, "type", None)
            content = getattr(m, "content", None)
        if not content:
            continue
        label = "Candidate" if speaker in ("human", "user") else "Coach"
        turns.append(f"{label}: {content}")

    if not turns:
        return "(no prior conversation yet — this is the first message)"

    if len(turns) <= keep_head + keep_tail:
        return "\n".join(turns)

    head = turns[:keep_head]
    tail = turns[-keep_tail:]
    omitted = len(turns) - keep_head - keep_tail
    return "\n".join(
        head
        + [f"... [{omitted} earlier message(s) omitted for length] ..."]
        + tail
    )


class state(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    classifier: str
    company_name: str
    role: str
    difficulty_level: str
    # Which DSA topic(s) this turn actually needs. Set fresh every turn by
    # reviwer_node (NOT merged) so a later turn never gets confused by
    # candidate_questions keys left over from an earlier turn.
    topic: str
    requested_topics: list
    final_result: str
    candidate_questions: Annotated[dict, merge_dict]


def build_RAG(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embedings)
    return vector_store.as_retriever(search_kwargs={"k": 4})


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# TODO: update this path to your actual resume PDF location
resume_retriever = build_RAG(
    os.path.join(BASE_DIR, "Hingu_Krishn_Resume_compressed.pdf")
)

# --- nodes ---

def classifier_node(state: state) -> dict:
    """looks at the user's input and decides which of the 5 branches to route to"""
    user_message = state['user_input']
    history = format_history(state)

    prompt = f"""You are a query classifier for an AI interview prep coach.

Recent conversation so far (most recent last):
{history}

Read the candidate's newest message below and classify it into EXACTLY ONE of the
following categories. Respond with only the category name in lowercase, with no
punctuation, quotes, explanation, or extra words.

Categories:
- dsa: The candidate wants you to GENERATE a brand-new coding/DSA practice question
  right now, AND names a specific topic (arrays/hashing, trees/graphs, or dynamic
  programming) — either in this message, or by clearly continuing a topic already
  established earlier in the conversation (e.g. "give me a harder one" right after
  a trees/graphs question was just given). Do NOT use "dsa" for a bare request with
  no topic and no established topic to continue (see rule below — that's
  general_chat instead). Do NOT use "dsa" for requests about study resources,
  links, general advice, or tips about DSA either — those are general_chat too.
- behavioral: The candidate gives a DIRECT, explicit command to generate a
  behavioral/HR-style question right now, AND either names a specific theme/focus
  (e.g. teamwork, conflict, leadership, handling failure, prioritization) or is
  clearly continuing a theme already established earlier in the conversation. A
  direct command with NO theme at all (e.g. "give me an hr question", "i would
  like to have hr questions", "ask me a behavioral question") is NOT enough by
  itself — see the rule below, that's general_chat instead so the coach can ask
  what theme they want. Do NOT use "behavioral" for a candidate merely musing,
  worrying, or stating an intention about behavioral/HR prep either.
- company_research: The candidate is asking about a specific company's interview
  process, culture, values, or recent news relevant to interviewing there.
- resume_gap: The candidate is asking what they should brush up on, or how their
  resume compares to what's expected for the target role.
- general_chat: Greetings, small talk, thanks, farewells, requests for study
  resources/links/advice, follow-up questions about something already discussed,
  corrections or clarifications, bare/underspecified requests that need a
  clarifying question first (see rules below), musings or statements of intent
  that aren't direct commands, or anything that doesn't clearly fit the categories
  above. When in doubt, choose general_chat rather than guessing.

IMPORTANT — use the conversation history above to read the newest message in
context, not in isolation:
- A short correction or clarification about what the candidate just asked (e.g.
  "no it was about hr question", "I meant something else") is almost always
  general_chat, even if it happens to mention a topic like "hr" or "dsa" —
  the candidate is fixing a misunderstanding, not requesting a brand-new question.
- A question referring back to something earlier in the conversation ("what did I
  ask you before", "go back to my last question") is general_chat, not a request
  to generate new content.
- A candidate CHECKING IN on something they believe already happened or was
  promised (e.g. "what about the hr question I told you to ask", "did you ever
  give me that DSA question", "weren't you going to ask me something on trees")
  is general_chat, NOT a request to generate a new one — even though it names a
  topic like "hr" or "dsa". Only route to that topic's category if the phrasing is
  a clear, direct ask for something new right now (e.g. "ask me one now").
- BARE DSA REQUESTS: if the candidate asks for a DSA question WITHOUT naming a
  topic (arrays/hashing, trees/graphs, dp), and no topic was already established
  earlier in the conversation to continue, that is general_chat — the coach needs
  to ask which topic and difficulty first, not guess one for them. Example: "i
  would like to have a dsa question" with no prior topic in the conversation ->
  general_chat, NOT dsa.
- BARE OR VAGUE BEHAVIORAL REQUESTS: if the candidate wants a behavioral/HR
  question but names NO theme — whether it's a direct command with no theme
  ("give me an hr question", "i would like to have hr questions", "ask me a
  behavioral question") or a vague musing/worry about needing practice ("i think
  i have to practice the hr round questions") — and no theme was already
  established earlier in the conversation to continue, that is general_chat, so
  the coach can ask what theme they'd like (or offer to pick one) before
  generating anything. Only route to "behavioral" once a theme is given or
  clearly implied by context.
- AFFIRMATIVE FOLLOW-THROUGH: if the coach's last message (see history above)
  offered to generate a question and asked for confirmation or missing details
  (topic/difficulty/theme), and the candidate's new message supplies that
  confirmation or those details (e.g. "yes", "sure, arrays please", "medium",
  "go ahead, something on teamwork"), treat that as the direct request now and
  route to the matching category (dsa or behavioral) — don't send it back to
  general_chat again.

Examples:
"give me a question on arrays" -> dsa
"ask me something on graphs" -> dsa
"give me an easier one" (right after a dsa question was just given) -> dsa
"i would like to have a dsa question" (no topic named, none established yet) -> general_chat
"give me a hard DSA question" (no topic named) -> general_chat
"give me a resource to learn DSA" -> general_chat
"how should I prepare for DSA rounds in general?" -> general_chat
"any tips for arrays and hashing?" -> general_chat
"what's it like interviewing at Google?" -> company_research
"what should I review before my SWE interview?" -> resume_gap
"ask me a question about handling conflict" -> behavioral
"give me an hr question about a time you failed" -> behavioral
"give me an hr question" (no theme named) -> general_chat
"i would like to have hr questions" (no theme named) -> general_chat
"i think i have to practice the hr round questions" -> general_chat
"i'm nervous about the behavioral round" -> general_chat
"which question did I ask you at the very beginning?" -> general_chat
"no, it was about the hr question" (correcting the coach's last reply) -> general_chat
"what about the hr question i told you to ask in the beginning" -> general_chat
"did you already give me a DSA question?" -> general_chat
"actually can you give me a fresh hr question instead" -> behavioral
"sure, arrays please, medium" (right after coach asked which topic/difficulty) -> dsa
"yes go ahead, something on teamwork" (right after coach asked if they want a behavioral question) -> behavioral

Candidate's newest message:
\"\"\"{user_message}\"\"\"

Category (one word, lowercase, nothing else):"""

    response = safe_llm_invoke(prompt)
    category = response.content.strip().lower()

    valid_categories = ["dsa", "behavioral", "company_research", "resume_gap", "general_chat"]
    if category not in valid_categories:
        category = "general_chat"

    return {
        "classifier": category,
        "messages": [("human", user_message)],
    }


def reviwer_node(state: state) -> dict:
    """Extracts BOTH the difficulty level (easy/medium/hard) AND the DSA topic
    from the candidate's message, in a single Groq call.

    Determining the topic here (instead of always fanning out to all three
    topic-generator nodes) is what lets router_function send the request
    straight to the one matching node most of the time — since classifier_node
    already only routes here when a topic is named or clearly continued from
    earlier in the conversation, "ambiguous" should be rare in practice.
    """
    user_message = state['user_input']
    history = format_history(state)

    prompt = f"""You extract two things from a candidate's DSA practice-question
request: the difficulty level and the topic area.

Recent conversation so far (most recent last), for resolving references like
"give me a harder one" or continuing whatever topic was already established:
{history}

Respond with ONLY a compact JSON object, nothing else — no markdown fences, no
explanation:
{{"topic": "<one of: arrays_hashing, trees_graphs, dp, ambiguous>", "difficulty": "<one of: easy, medium, hard>"}}

Rules for "topic":
- arrays_hashing: arrays, hashing, hash maps/sets, two pointers, sliding window,
  prefix sums, strings.
- trees_graphs: binary trees, BSTs, general trees, graphs, BFS/DFS, union-find,
  topological sort.
- dp: dynamic programming, memoization, tabulation.
- If the newest message doesn't name a topic, but a topic was clearly already
  established earlier in the conversation and this message is continuing it
  (e.g. "give me a harder one" right after a trees/graphs question), use that
  established topic.
- If you genuinely cannot determine a single topic (no topic named now, and none
  established to continue), use "ambiguous".

Rules for "difficulty":
- If the message explicitly names a difficulty ("easy", "medium", "hard", or
  close synonyms like "simple"/"basic" -> easy, "tough"/"challenging"/"tricky" ->
  hard), use that.
- If the message asks for something relative to a prior question ("harder one",
  "step it up") -> "hard". If it asks for something easier -> "easy".
- If no difficulty is stated or implied at all, default to "medium".

Candidate's newest message:
\"\"\"{user_message}\"\"\"

JSON:"""

    response = safe_llm_invoke(prompt)
    raw = response.content.strip()
    # Strip accidental markdown fences before parsing.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip() if raw.lower().startswith("json") else raw

    topic = "ambiguous"
    difficulty = "medium"
    try:
        data = json.loads(raw)
        topic = str(data.get("topic", "ambiguous")).strip().lower()
        difficulty = str(data.get("difficulty", "medium")).strip().lower()
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("reviwer_node: couldn't parse JSON (%s), raw was: %r", e, raw)

    valid_topics = ["arrays_hashing", "trees_graphs", "dp"]
    if topic not in valid_topics:
        topic = "ambiguous"

    valid_difficulties = ["easy", "medium", "hard"]
    if difficulty not in valid_difficulties:
        difficulty = "medium"

    requested_topics = [topic] if topic != "ambiguous" else list(valid_topics)

    return {
        "difficulty_level": difficulty,
        "topic": topic,
        "requested_topics": requested_topics,
    }


def array_hasing_node(state: state) -> dict:
    """asking questions about array_hasing_node topic for DSA"""
    difficulty_level = state['difficulty_level']

    prompt = f"""You are an experienced technical interviewer creating a DSA practice
question for a candidate.

Generate exactly ONE original interview question on the topic of arrays and hashing,
calibrated to a {difficulty_level} difficulty level for a real software engineering
interview.

Guidelines for calibration:
- easy: a single well-known pattern (e.g. two-sum style), solvable in O(n) with basic
  hashing, minimal edge cases.
- medium: requires combining hashing with another technique (sliding window, prefix
  sums, sorting) or handling multiple edge cases.
- hard: requires a non-obvious insight, tight complexity constraints, or combining
  hashing with a less common data structure/trick.

CRITICAL INSTRUCTIONS:
1. Output ONLY the question text itself — nothing else.
2. Do not include the answer, hints, approach, or explanation.
3. Do not add labels like "Question:" or any markdown formatting.
4. Phrase it exactly as it would be asked out loud in a real interview, including any
   necessary constraints (input size, value ranges) if relevant to the difficulty.
5. Avoid the most overused textbook example (e.g. the exact classic "two sum" wording)
   unless the difficulty is easy and no fresher equivalent fits as well.
"""
    response = safe_llm_invoke(prompt)
    return {"candidate_questions": {"arrays_hashing": response.content.strip()}}


def trees_graphs_node(state: state) -> dict:
    """asking questions about trees_graphs_node topic for DSA"""
    difficulty_level = state['difficulty_level']

    prompt = f"""You are an experienced technical interviewer creating a DSA practice
question for a candidate.

Generate exactly ONE original interview question on the topic of trees and graphs,
calibrated to a {difficulty_level} difficulty level for a real software engineering
interview.

Guidelines for calibration:
- easy: a single standard traversal (BFS/DFS) or basic tree property check.
- medium: requires combining traversal with additional logic (e.g. shortest path,
  level-aware processing, tree reconstruction, cycle detection).
- hard: requires an advanced algorithm (e.g. topological sort with constraints,
  union-find, LCA, or non-trivial state tracking during traversal).

CRITICAL INSTRUCTIONS:
1. Output ONLY the question text itself — nothing else.
2. Do not include the answer, hints, approach, or explanation.
3. Do not add labels like "Question:" or any markdown formatting.
4. Phrase it exactly as it would be asked out loud in a real interview, including any
   necessary constraints (graph size, directed/undirected, weighted/unweighted) if
   relevant to the difficulty.
5. Avoid the most overused textbook example unless the difficulty is easy and no
   fresher equivalent fits as well.
"""
    response = safe_llm_invoke(prompt)
    return {"candidate_questions": {"trees_graphs": response.content.strip()}}


def dp_node(state: state) -> dict:
    """asking questions about dp_node topic for DSA"""
    difficulty_level = state['difficulty_level']

    prompt = f"""You are an experienced technical interviewer creating a DSA practice
question for a candidate.

Generate exactly ONE original interview question on the topic of dynamic programming,
calibrated to a {difficulty_level} difficulty level for a real software engineering
interview.

Guidelines for calibration:
- easy: a classic 1D DP with an obvious recurrence (e.g. climbing stairs style).
- medium: requires identifying a less obvious state (2D DP, DP over strings/arrays
  with an extra dimension) or optimizing space.
- hard: requires a non-obvious state definition, multiple constraints combined, or
  DP paired with another technique (bitmask, binary search on the answer, graph DP).

CRITICAL INSTRUCTIONS:
1. Output ONLY the question text itself — nothing else.
2. Do not include the answer, hints, approach, or explanation.
3. Do not add labels like "Question:" or any markdown formatting.
4. Phrase it exactly as it would be asked out loud in a real interview, including any
   necessary constraints (input size, ranges) if relevant to the difficulty.
5. Avoid the most overused textbook example unless the difficulty is easy and no
   fresher equivalent fits as well.
"""
    response = safe_llm_invoke(prompt)
    return {"candidate_questions": {"dp": response.content.strip()}}


def picker_node(state: state) -> dict:
    """Picks the single best candidate question.

    candidate_questions is merged (never cleared) across the whole session by
    design, so it can still contain keys from earlier turns. requested_topics
    is set FRESH every turn by reviwer_node, so we filter down to just this
    turn's candidates before deciding anything — that's what keeps an old
    "trees_graphs" question from a previous turn out of today's decision.

    When only one topic was actually requested (the common case now that
    routing goes straight to the matching node), there's nothing to pick
    between — skip the LLM call entirely and return that candidate directly.
    """
    user_message = state['user_input']
    difficulty_level = state['difficulty_level']
    requested_topics = state.get('requested_topics') or list(state['candidate_questions'].keys())
    all_candidates = state['candidate_questions']

    candidates = {k: v for k, v in all_candidates.items() if k in requested_topics}
    if not candidates:
        # Defensive fallback — should not normally happen.
        candidates = all_candidates

    if len(candidates) == 1:
        final_question = next(iter(candidates.values()))
        return {
            "final_result": final_question,
            "messages": [("ai", final_question)],
        }

    topic_keys = list(candidates.keys())
    candidates_text = "\n\n".join(
        [f"[{topic}]\n{question}" for topic, question in candidates.items()]
    )

    prompt = f"""You are helping select the single best DSA interview question to show
a candidate, out of several already-generated options.

The candidate's original request was:
\"\"\"{user_message}\"\"\"

All candidates below were generated at the {difficulty_level} difficulty level. Each
is labeled with its topic key in square brackets.

Candidates:
{candidates_text}

CRITICAL INSTRUCTIONS:
1. If the candidate's request clearly names or implies a specific topic (e.g. "array
   question", "something on graphs", "a DP problem"), pick the matching candidate for
   that topic — even if you think another one reads slightly better.
2. If no topic preference is stated or implied, pick whichever candidate is clearest,
   most realistic, and best matches the {difficulty_level} difficulty level.
3. Respond with ONLY the exact topic key of your chosen candidate, copied verbatim
   from the square brackets above (one of: {", ".join(topic_keys)}).
4. Output nothing else — no explanation, no punctuation, no quotes, no extra text.

Chosen topic key:"""
    response = safe_llm_invoke(prompt)

    chosen_topic = response.content.strip().split("\n")[0].strip().lower()
    final_question = candidates.get(chosen_topic, list(candidates.values())[0])

    return {
        "final_result": final_question,
        "messages": [("ai", final_question)],
    }


def company_reasearch_node(state: state) -> dict:
    """search for a company's interview process, culture, recent news"""
    company_name = state['company_name']
    role = state['role']

    search_query = f"{company_name} {role} interview process culture recent news"
    search_results = search_tool.invoke(search_query)

    prompt = f"""You are a career coach helping a candidate prepare for an interview at
{company_name} for the role of {role}.

Use ONLY the search results below to write a concise, practical prep briefing.

CRITICAL INSTRUCTIONS:
1. Base every claim strictly on the search results provided. Never invent interview
   stages, culture claims, or news that isn't actually supported by them.
2. If the search results don't cover a section (e.g. no interview-process detail was
   found), say so plainly in that section instead of guessing or padding with generic
   advice.
3. Structure the answer under exactly these three headers, in this order:
   - Interview Process
   - Culture
   - Recent News
4. Under each header, use short bullet points (2-4) rather than long paragraphs.
5. End with a single "Why this matters" line connecting the most actionable detail to
   how the candidate should prepare — only if the search results support it.
6. Keep the whole answer tight and skimmable; no filler sentences.

company:
{company_name}

role:
{role}

search results:
{search_results}
"""
    response = safe_llm_invoke(prompt)
    final_result = response.content.strip()
    return {
        "final_result": final_result,
        "messages": [("ai", final_result)],
    }


def resume_gap_node(state: state) -> dict:
    """identifies the gap in the resumes"""
    role = state['role']
    company_name = state['company_name']
    user_message = state['user_input']

    query = f"skills, experience, and projects relevant to a {role} position"
    retrieved_docs = resume_retriever.invoke(query)
    resume_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""You are a career coach helping a candidate prepare for a job interview
for {role} at {company_name}.

Compare the resume excerpts below against what is typically expected for this role,
and identify concrete gaps the candidate should be ready to address or brush up on.

CRITICAL INSTRUCTIONS:
1. Base your analysis strictly on the resume content provided below. Never invent
   skills, projects, or experience that aren't actually present in it.
2. If the retrieved resume context looks thin, unrelated to the role, or empty, say so
   plainly instead of fabricating an analysis.
3. Be specific — name exact skills, tools, or experience areas that are missing or
   weak for this role. Avoid vague statements like "needs more experience."
4. Also explicitly call out what IS already strong on the resume for this role, so the
   answer isn't purely critical.
5. Keep the tone encouraging and constructive, like a mentor helping them prepare —
   never harshly critical.
6. Structure the answer as: a short "Strengths" section, then a short "Gaps to
   address" section, each 2-4 bullet points.
7. Keep the whole answer under 200 words.

role:
{role} at {company_name}

resume context:
\"\"\"{resume_context}\"\"\"

candidate's question:
\"\"\"{user_message}\"\"\"
"""
    response = safe_llm_invoke(prompt)
    final_result = response.content.strip()
    return {
        "final_result": final_result,
        "messages": [("ai", final_result)],
    }


def behavioral_node(state: state) -> dict:
    """generates a behavioral / HR-style interview question tailored to the role"""
    role = state['role']
    company_name = state['company_name']
    user_message = state['user_input']

    prompt = f"""You are an experienced interviewer creating a behavioral / HR-style
interview question for a candidate interviewing for {role} at {company_name}.

CRITICAL INSTRUCTIONS:
1. Base the question on a theme genuinely relevant to this specific role (e.g.
   teamwork, conflict resolution, ownership, handling failure, prioritization,
   ambiguity, cross-functional communication) — pick the theme that best fits a
   {role} rather than defaulting to the most generic option.
2. If the candidate's request below specifies a theme, honor it.
3. Ground the question in a concrete, specific scenario (a tool, a deadline, a team
   dynamic, a decision under pressure) rather than staying abstract — avoid the most
   generic phrasing for the theme (e.g. plain "tell me about a weakness") in favor of
   a sharper, more particular version of the same idea.
4. Output ONLY the question text itself, phrased the way it would actually be asked
   out loud (e.g. "Tell me about a time when...").
5. Do not include the answer, tips, the STAR method explanation, or any follow-up
   prompts.
6. Do not add labels like "Question:" or any markdown formatting.

candidate's request:
\"\"\"{user_message}\"\"\"
"""
    response = safe_llm_invoke(prompt)
    final_result = response.content.strip()
    return {
        "final_result": final_result,
        "messages": [("ai", final_result)],
    }


def general_chat_node(state: state) -> dict:
    """handles greetings, small talk, and anything that isn't a specific coaching request"""
    role = state.get('role', '')
    company_name = state.get('company_name', '')
    # Includes the candidate's newest message as the final line, since
    # classifier_node already records it before routing here.
    history = format_history(state)

    prompt = f"""You are a friendly, sharp AI interview prep coach chatting casually
with a candidate preparing for a {role} role at {company_name}.

Actual conversation so far, in order (most recent last — the last "Candidate" line
is the message you're responding to right now):
{history}

Respond naturally and briefly, like a real person texting back — never like a
scripted assistant reciting a feature list.

CRITICAL — grounding in real history:
If the candidate asks about anything from earlier in the conversation (e.g. "what
did I ask first", "what was your last answer", "which question did you give me
before"), answer using ONLY what actually appears in the conversation above. Quote
or accurately summarize the real prior message — never invent, guess, or restate
the candidate's current question back to them as if it were the answer. If the
history above genuinely doesn't contain what they're asking about, say so honestly
instead of making something up.

If the candidate's message is a correction or clarification (e.g. "no, I meant X",
"that's not what I asked"), acknowledge the correction directly and address what
they actually meant — don't ignore it and go generate something unrelated.

If the candidate is CHECKING IN on something they believe already happened or was
promised (e.g. "what about the hr question I told you to ask", "did you already
give me a DSA question"):
- If it's actually in the conversation above, point to it directly — quote or
  closely paraphrase the real thing so they can find it, rather than just saying
  "yes I did."
- If it genuinely is NOT in the conversation above, say so plainly and ask a short
  clarifying question about what they'd like next (e.g. "I don't see an HR
  question in our chat yet — want me to give you one now?"). Do not silently
  generate a new question yourself here; that decision belongs to the candidate,
  and jumping straight to content when they were just checking in is exactly the
  kind of unrequested action to avoid.

If the candidate asked for a DSA question WITHOUT naming a topic (and no topic was
already established earlier to continue), do NOT generate or describe a question
yourself. Just ask, in one short friendly line, which topic they'd like — arrays &
hashing, trees & graphs, or dynamic programming — and whether they want it easy,
medium, or hard. Nothing else.

If the candidate wants a behavioral/HR question but hasn't named a theme —
whether that's a direct request with no theme ("give me an hr question", "i would
like to have hr questions") or a musing/worry about needing practice ("i think i
have to practice the hr round questions") — do NOT generate a question yourself.
Acknowledge what they said in one short line, then ask whether there's a
particular theme (teamwork, conflict, ownership, handling failure, prioritization,
etc.) they want to focus on — or say you'll pick one for them if they don't have a
preference.

Read the room otherwise:
- If they're greeting you for the first time, or clearly don't know what you can do,
  you may briefly mention you can help with DSA questions, behavioral questions,
  company research, or resume gap analysis — but only ONCE in this conversation, and
  only if it fits naturally in one clause, not a bulleted pitch.
- If they're asking for general study resources, links, or advice (not a fresh
  practice question), give a couple of genuinely useful, specific suggestions rather
  than a generic "here are some resources" brush-off.
- If they're saying bye, thanks, or wrapping up, respond warmly and briefly. Do NOT
  pitch your features again or ask how their prep is going if that's already come up
  earlier in this conversation.
- If they're just chatting (a joke, small talk, a random comment), just chat back in
  kind. Don't force a redirect toward interview prep every time.
- Never repeat a phrase, question, or feature pitch you've already used earlier in
  this conversation.

Keep it to 1-2 sentences. No bullet points, no lists, no repeated sign-offs.
"""
    response = safe_llm_invoke(prompt)
    final_result = response.content.strip()
    return {
        "final_result": final_result,
        "messages": [("ai", final_result)],
    }


def router_function(state: state):
    """reads the category set by classifier_node and returns the ACTUAL node name to go to next"""
    category = state['classifier']
    if category == "dsa":
        return "reviwer_node"
    elif category == "company_research":
        return "company_reasearch_node"
    elif category == "resume_gap":
        return "resume_gap_node"
    elif category == "behavioral":
        return "behavioral_node"
    else:
        return "general_chat_node"


def topic_router_function(state: state):
    """Reads the topic set by reviwer_node and routes to JUST the matching
    topic-generator node — instead of always firing all three in parallel.

    All three Groq calls hitting at once was the actual cause of the
    intermittent 400s: a burst like that is an easy way to trip a
    provider-side rate limit even under otherwise-light traffic. Now that
    reviwer_node determines the topic, we only pay for a 3-way fan-out on
    the (should be rare) "ambiguous" case — and even then, safe_llm_invoke
    will retry through a transient 429 instead of failing the whole request.
    """
    topic = state.get('topic', 'ambiguous')
    if topic == "arrays_hashing":
        return ["array_hasing_node"]
    elif topic == "trees_graphs":
        return ["trees_graphs_node"]
    elif topic == "dp":
        return ["dp_node"]
    else:
        return ["array_hasing_node", "trees_graphs_node", "dp_node"]

#  it will called once when the session created 
# of the create graph we need to wrap it into the function and save it memorysaver

def create_graph():
    """Builds and compiles the graph WITH a checkpointer, so each thread_id
       (i.e. each user session) gets its own persisted state automatically."""

    graph = StateGraph(state)

    graph.add_node("classifier_node", classifier_node)
    graph.add_node("reviwer_node", reviwer_node)
    graph.add_node("array_hasing_node", array_hasing_node)
    graph.add_node("trees_graphs_node", trees_graphs_node)
    graph.add_node("dp_node", dp_node)
    graph.add_node("picker_node", picker_node)
    graph.add_node("company_reasearch_node", company_reasearch_node)
    graph.add_node("resume_gap_node", resume_gap_node)
    graph.add_node("behavioral_node", behavioral_node)
    graph.add_node(general_chat_node)
    graph.add_edge(START, "classifier_node")

    graph.add_conditional_edges("classifier_node", router_function)
    graph.add_conditional_edges(
        "reviwer_node",
        topic_router_function,
        ["array_hasing_node", "trees_graphs_node", "dp_node"],
    )

    graph.add_edge("array_hasing_node", "picker_node")
    graph.add_edge("trees_graphs_node", "picker_node")
    graph.add_edge("dp_node", "picker_node")

    graph.add_edge("picker_node", END)
    graph.add_edge("company_reasearch_node", END)
    graph.add_edge("resume_gap_node", END)
    graph.add_edge("behavioral_node", END)
    graph.add_edge("general_chat_node", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)