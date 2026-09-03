import os
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

# --- shared setup (loaded once when this module is imported) ---

search_tool = TavilySearch(max_results=3)
tools = [search_tool]

llm =  ChatGroq(model="openai/gpt-oss-120b" , temperature=0.2)
llm_tools = llm.bind_tools(tools)

embedings = LightHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def merge_dict(left, right):
    """Custom reducer to merge candidate questions without overwriting."""
    if left is None:
        return right
    if right is None:
        return left
    merged = left.copy()
    merged.update(right)
    return merged


class state(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    classifier: str
    company_name: str
    role: str
    difficulty_level: str
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

    prompt = f"""You are a query classifier for an AI interview prep coach.

Read the candidate's message and classify it into EXACTLY ONE of the following
categories. Respond with only the category name in lowercase, with no punctuation,
quotes, explanation, or extra words.

Categories:
- dsa: The user wants you to GENERATE a brand-new coding/DSA practice question for
  them to solve right now (e.g. "give me a question", "ask me something on trees",
  "give me a harder one", "another one please"). Do NOT use "dsa" for requests about
  study resources, links, general advice, or tips about DSA — route those to
  "general_chat" instead.
- behavioral: The user wants a behavioral or HR-style interview question generated.
- company_research: The user is asking about a specific company's interview process,
  culture, values, or recent news relevant to interviewing there.
- resume_gap: The user is asking what they should brush up on, or how their resume
  compares to what's expected for the target role.
- general_chat: Greetings, small talk, thanks, farewells, requests for study
  resources/links/advice, follow-up questions about a question already given (e.g.
  "can you explain that answer"), or anything that doesn't clearly fit the categories
  above. When in doubt, choose general_chat rather than guessing.

Examples:
"give me a hard DSA question" -> dsa
"ask me something on graphs" -> dsa
"give me an easier one" -> dsa
"give me a resource to learn DSA" -> general_chat
"how should I prepare for DSA rounds in general?" -> general_chat
"any tips for arrays and hashing?" -> general_chat
"what's it like interviewing at Google?" -> company_research
"what should I review before my SWE interview?" -> resume_gap
"ask me a question about handling conflict" -> behavioral

Candidate message:
\"\"\"{user_message}\"\"\"

Category (one word, lowercase, nothing else):"""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    valid_categories = ["dsa", "behavioral", "company_research", "resume_gap", "general_chat"]
    if category not in valid_categories:
        category = "general_chat"

    return {"classifier": category}


def reviwer_node(state: state) -> dict:
    """detects the difficulty level (hard/medium/easy) the user asked for"""
    user_message = state['user_input']

    prompt = f"""You extract the requested difficulty level from a candidate's message
about a DSA practice question.

Respond with EXACTLY ONE word — "easy", "medium", or "hard" — and nothing else: no
punctuation, no explanation, no quotes.

Rules:
- If the message explicitly names a difficulty ("easy", "medium", "hard", or close
  synonyms like "simple"/"basic" -> easy, "tough"/"challenging"/"tricky" -> hard),
  return that difficulty.
- If the message asks for something relative to a prior question ("harder one",
  "step it up") -> return "hard". If it asks for something easier/simpler than
  before -> return "easy".
- If no difficulty is stated or implied at all, default to "medium".

Candidate message:
\"\"\"{user_message}\"\"\"

Difficulty (one word only):"""

    response = llm.invoke(prompt)
    difficulty = response.content.strip().lower()

    valid_difficulties = ["easy", "medium", "hard"]
    if difficulty not in valid_difficulties:
        difficulty = "medium"

    return {"difficulty_level": difficulty}


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
    response = llm.invoke(prompt)
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
    response = llm.invoke(prompt)
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
    response = llm.invoke(prompt)
    return {"candidate_questions": {"dp": response.content.strip()}}


def picker_node(state: state) -> dict:
    """picks the single best candidate question out of the 3 parallel candidates"""
    user_message = state['user_input']
    difficulty_level = state['difficulty_level']
    candidates = state['candidate_questions']

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
    response = llm.invoke(prompt)

    chosen_topic = response.content.strip().split("\n")[0].strip().lower()
    final_question = candidates.get(chosen_topic, list(candidates.values())[0])

    return {"final_result": final_question}


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
    response = llm.invoke(prompt)
    return {"final_result": response.content.strip()}


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
    response = llm.invoke(prompt)
    return {"final_result": response.content.strip()}


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
2. If the candidate's request below specifies a theme or scenario, honor it.
3. Output ONLY the question text itself, phrased the way it would actually be asked
   out loud (e.g. "Tell me about a time when...").
4. Do not include the answer, tips, the STAR method explanation, or any follow-up
   prompts.
5. Do not add labels like "Question:" or any markdown formatting.

candidate's request:
\"\"\"{user_message}\"\"\"
"""
    response = llm.invoke(prompt)
    return {"final_result": response.content.strip()}

def general_chat_node(state: state) -> dict:
    """handles greetings, small talk, and anything that isn't a specific coaching request"""
    user_message = state['user_input']
    role = state.get('role', '')
    company_name = state.get('company_name', '')
    prompt = f"""You are a friendly, sharp AI interview prep coach chatting casually
with a candidate preparing for a {role} role at {company_name}.

The candidate just said:
\"\"\"{user_message}\"\"\"

Respond naturally and briefly, like a real person texting back — never like a
scripted assistant reciting a feature list.

Read the room:
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
    response = llm.invoke(prompt)
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
    graph.add_edge("reviwer_node", "array_hasing_node")
    graph.add_edge("reviwer_node", "trees_graphs_node")
    graph.add_edge("reviwer_node", "dp_node")

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