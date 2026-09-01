import os
import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# ============================================================
# Graph logic (same nodes as your CLI version)
# ============================================================

def merge_dict(left, right):
    """Custom reducer to merge candidate questions without overwriting."""
    if left is None:
        return right
    if right is None:
        return left
    merged = left.copy()
    merged.update(right)
    return merged


def append_list(left, right):
    """Custom reducer to append asked questions without overwriting past ones."""
    return (left or []) + (right or [])


load_dotenv()


class state(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    classifier: str
    company_name: str
    role: str
    difficulty_level: str
    final_result: str
    candidate_questions: Annotated[dict, merge_dict]
    asked_questions: Annotated[list, append_list]


@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)


@st.cache_resource
def get_creative_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)


@st.cache_resource
def get_search_tool():
    return TavilySearch(max_results=3)


@st.cache_resource
def get_resume_retriever(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embedings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embedings)
        return vector_store.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.warning(f"Could not load resume for RAG: {e}")
        return None


llm = get_llm()
creative_llm = get_creative_llm()
search_tool = get_search_tool()

# NOTE: update this path to match wherever the resume actually lives,
# or point a RESUME_PATH env var at it
RESUME_PATH = os.environ.get("RESUME_PATH", r"conditional-workflow\Hingu_Krishn_Resume_compressed.pdf")
resume_retriever = get_resume_retriever(RESUME_PATH)


def classifier_node(state: state) -> dict:
    """routes the user's question into dsa / behavioral / company_research / resume_gap / general_chat"""
    user_message = state['user_input']

    prompt = f"""Classify the user's request into exactly one category:
    dsa, behavioral, company_research, resume_gap, general_chat

    Reply with ONLY the category word, nothing else.

    Rules:
    - "dsa": wants a coding/DSA practice question
    - "behavioral": wants a behavioral/HR-style question
    - "company_research": wants to know about a company's interview process, culture, or news
    - "resume_gap": wants to know what to brush up on based on their resume vs the role
    - "general_chat": greetings, small talk, thanks, or anything not specifically
      requesting a practice question or research

    user_input:
    {user_message}
    """

    response = llm.invoke(prompt)

    return {
        "classifier": response.content.strip().lower(),
        "messages": [("human", user_message)],
    }


def reviwer_node(state: state) -> dict:
    """detects the difficulty level (hard/medium/easy) the user asked for"""
    user_message = state['user_input']
    previous_difficulty = state.get('difficulty_level', 'medium')

    prompt = f"previous difficulty level was {previous_difficulty}. \
            identify the difficulty the user wants next from user_input \
            if user_input mentions a difficulty directly (easy, medium, hard) use that \
            if user_input asks for something relative like easier or harder, \
            move one step from {previous_difficulty} in that direction \
            if user_input mentions nothing about difficulty, keep {previous_difficulty} \
            only give me one word answer: easy, medium, or hard \
            user_input \
            {user_message}"

    response = llm.invoke(prompt)
    difficulty = response.content.strip().lower()

    if difficulty not in ("easy", "medium", "hard"):
        difficulty = previous_difficulty

    return {"difficulty_level": difficulty}


def array_hasing_node(state: state) -> dict:
    """asking questions about arrays/hashing topic for DSA"""
    difficulty_level = state['difficulty_level']
    already_asked = state.get('asked_questions', [])
    avoid_text = "\n".join(already_asked) if already_asked else "none yet"

    prompt = f"""You are an interviewer creating a DSA practice question.

    Generate exactly ONE interview question on the topic of arrays and hashing,
    at a {difficulty_level} difficulty level.

    Avoid generating a question similar to any of these already asked:
    {avoid_text}

    CRITICAL INSTRUCTIONS:
    1. Output ONLY the question text itself, nothing else.
    2. Do not include the answer, hints, or explanation.
    3. Do not add labels like "Question:" or markdown formatting.
    4. Keep it realistic, the way it would actually be asked in a real interview.
    """
    response = creative_llm.invoke(prompt)
    return {"candidate_questions": {"arrays_hashing": response.content.strip()}}


def trees_graphs_node(state: state) -> dict:
    """asking questions about trees/graphs topic for DSA"""
    difficulty_level = state['difficulty_level']
    already_asked = state.get('asked_questions', [])
    avoid_text = "\n".join(already_asked) if already_asked else "none yet"

    prompt = f"""You are an interviewer creating a DSA practice question.

    Generate exactly ONE interview question on the topic of trees and graphs,
    at a {difficulty_level} difficulty level.

    Avoid generating a question similar to any of these already asked:
    {avoid_text}

    CRITICAL INSTRUCTIONS:
    1. Output ONLY the question text itself, nothing else.
    2. Do not include the answer, hints, or explanation.
    3. Do not add labels like "Question:" or markdown formatting.
    4. Keep it realistic, the way it would actually be asked in a real interview.
    """
    response = creative_llm.invoke(prompt)
    return {"candidate_questions": {"trees_graphs": response.content.strip()}}


def dp_node(state: state) -> dict:
    """asking questions about dynamic programming topic for DSA"""
    difficulty_level = state['difficulty_level']
    already_asked = state.get('asked_questions', [])
    avoid_text = "\n".join(already_asked) if already_asked else "none yet"

    prompt = f"""You are an interviewer creating a DSA practice question.

    Generate exactly ONE interview question on the topic of dynamic programming,
    at a {difficulty_level} difficulty level.

    Avoid generating a question similar to any of these already asked:
    {avoid_text}

    CRITICAL INSTRUCTIONS:
    1. Output ONLY the question text itself, nothing else.
    2. Do not include the answer, hints, or explanation.
    3. Do not add labels like "Question:" or markdown formatting.
    4. Keep it realistic, the way it would actually be asked in a real interview.
    """
    response = creative_llm.invoke(prompt)
    return {"candidate_questions": {"dp": response.content.strip()}}


def picker_node(state: state) -> dict:
    """picks the single best candidate question out of the 3 parallel candidates"""
    user_message = state['user_input']
    difficulty_level = state['difficulty_level']
    candidates = state['candidate_questions']

    candidates_text = "\n\n".join(
        [f"{topic}:\n{question}" for topic, question in candidates.items()]
    )

    prompt = f"""You are helping pick the best DSA interview question to show a candidate.

    Below are 3 candidate questions from different topics, all generated at the
    {difficulty_level} difficulty level. The candidate's original request was:
    "{user_message}"

    CRITICAL INSTRUCTIONS:
    1. If the candidate's request mentions a specific topic (like "array question" or
       "something on graphs"), pick that matching candidate.
    2. If no topic preference is mentioned, pick whichever candidate is clearest,
       most realistic, and best matches the {difficulty_level} difficulty level.
    3. Reply with ONLY the topic name of your chosen candidate on the first line
       (one of: {", ".join(candidates.keys())}).
    4. Do not add any other text on that first line.

    candidates:
    {candidates_text}
    """
    response = llm.invoke(prompt)

    chosen_topic = response.content.strip().split("\n")[0].strip().lower()

    if chosen_topic not in candidates:
        chosen_topic = next(iter(candidates))

    final_question = candidates[chosen_topic]

    return {
        "final_result": final_question,
        "messages": [("ai", final_question)],
        "asked_questions": [final_question],
    }


def general_chat_node(state: state) -> dict:
    """handles greetings, small talk, and anything that isn't a specific coaching request"""
    user_message = state['user_input']
    role = state.get('role', '')
    company_name = state.get('company_name', '')
    prompt = f"""You are a friendly AI interview prep coach chatting casually with a candidate
    preparing for a {role} role at {company_name}.


    The candidate just said:
    {user_message}

    Respond naturally and briefly — like a real person, not a scripted assistant.

    Read the room:
    - If they're greeting you for the first time or seem unsure what you do, you can
      mention you can help with DSA questions, behavioral questions, company research,
      or resume gap analysis — but only ONCE per conversation, and only if it fits naturally.
    - If they're saying bye, thanks, or wrapping up, just say something warm and short back.
      Do NOT pitch your features again or ask how their prep is going if you've already
      asked that earlier in this conversation.
    - If they're just chatting (a joke, small talk, a random comment), just chat back.
      Don't redirect every reply toward interview prep.
    - Never repeat a phrase or question you already used earlier in this conversation.

    Keep it to 1-2 sentences. No bullet points, no lists, no repeated sign-offs.
    """
    response = llm.invoke(prompt)
    final_result = response.content.strip()
    return {
        "final_result": final_result,
        "messages": [("ai", final_result)],
    }


def company_reasearch_node(state: state) -> dict:
    """search for a company's interview process, culture, recent news"""
    company_name = state['company_name']
    role = state['role']

    search_query = f"{company_name} {role} interview process culture recent news"

    try:
        search_results = search_tool.invoke(search_query)
    except Exception as e:
        search_results = f"(search failed: {e})"

    prompt = f"""You are a career coach helping a candidate prepare for an interview.

    Use the search results below to summarize what the candidate should know about the
    company's interview process, work culture, and any recent news relevant to the role.

    CRITICAL INSTRUCTIONS:
    1. Base your answer ONLY on the search results provided below. Do not invent details
       that aren't supported by them.
    2. If the search results don't cover something (e.g. interview process specifics),
       say so honestly instead of guessing.
    3. Organize the answer into short sections: Interview Process, Culture, Recent News.
    4. Keep it concise and practical, focused on what actually helps interview prep.

    company:
    {company_name}

    role:
    {role}

    search results:
    {search_results}
    """
    response = llm.invoke(prompt)
    result = response.content.strip()

    return {
        "final_result": result,
        "messages": [("ai", result)],
    }


def resume_gap_node(state: state) -> dict:
    """identifies the gaps in the resume relative to the target role"""
    role = state['role']
    company_name = state['company_name']

    if resume_retriever is None:
        result = "I couldn't load your resume for this comparison — please check the resume file path."
        return {"final_result": result, "messages": [("ai", result)]}

    query = f"skills, experience, and projects relevant to a {role} position"
    retrieved_docs = resume_retriever.invoke(query)
    resume_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""You are a career coach helping a candidate prepare for a job interview.

    Your task is to compare the candidate's resume against what is typically expected
    for the given role, and point out gaps they should be ready to address or brush up on.

    CRITICAL INSTRUCTIONS:
    1. Base your analysis ONLY on the resume content provided below. Do not invent skills,
       projects, or experience that aren't actually present in the resume.
    2. Be specific: name the exact skills or experience areas that are missing or weak for
       this role, not vague statements like "needs more experience".
    3. Also mention what IS already strong on the resume for this role, so the answer isn't
       purely critical.
    4. Keep the tone encouraging and constructive, like a mentor helping them prepare, not
       harshly critical.
    5. Keep the answer focused and practical, ideally under 200 words.

    role:
    {role} at {company_name}

    resume context:
    {resume_context}
    """
    response = llm.invoke(prompt)
    result = response.content.strip()

    return {
        "final_result": result,
        "messages": [("ai", result)],
    }


def behavioral_node(state: state) -> dict:
    """generates a behavioral / HR-style interview question tailored to the role"""
    role = state['role']
    company_name = state['company_name']
    user_message = state['user_input']

    prompt = f"""You are an interviewer creating a behavioral interview question.

    Generate exactly ONE behavioral / HR-style interview question suitable for a candidate
    interviewing for the role of {role} at {company_name}.

    CRITICAL INSTRUCTIONS:
    1. Base the question on common themes for this type of role (teamwork, conflict resolution,
       leadership, handling failure, prioritization, etc).
    2. Output ONLY the question text itself, nothing else.
    3. Do not include the answer, tips, or the STAR method explanation.
    4. Do not add labels like "Question:" or markdown formatting.

    candidate's request:
    {user_message}
    """
    response = llm.invoke(prompt)
    result = response.content.strip()

    return {
        "final_result": result,
        "messages": [("ai", result)],
    }


def route_from_classifier(state: state):
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


def fan_out_dsa(state: state):
    return ["array_hasing_node", "trees_graphs_node", "dp_node"]


@st.cache_resource
def build_graph():
    graph = StateGraph(state)

    graph.add_node(classifier_node)
    graph.add_node(reviwer_node)
    graph.add_node(array_hasing_node)
    graph.add_node(trees_graphs_node)
    graph.add_node(dp_node)
    graph.add_node(picker_node)
    graph.add_node(company_reasearch_node)
    graph.add_node(resume_gap_node)
    graph.add_node(behavioral_node)
    graph.add_node(general_chat_node)

    graph.add_edge(START, "classifier_node")
    graph.add_conditional_edges("classifier_node", route_from_classifier)
    graph.add_conditional_edges("reviwer_node", fan_out_dsa)

    graph.add_edge("array_hasing_node", "picker_node")
    graph.add_edge("trees_graphs_node", "picker_node")
    graph.add_edge("dp_node", "picker_node")

    graph.add_edge("picker_node", END)
    graph.add_edge("company_reasearch_node", END)
    graph.add_edge("resume_gap_node", END)
    graph.add_edge("behavioral_node", END)
    graph.add_edge("general_chat_node", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


app = build_graph()

# ============================================================
# Streamlit UI (basic)
# ============================================================

st.set_page_config(page_title="AI Interview Prep Coach", page_icon="🎯", layout="centered")

st.title("🎯 AI Interview Prep Coach")
st.caption("Built with LangGraph — classifier routing, parallel DSA fan-out, Tavily web search, and FAISS resume RAG.")

# --- Sidebar: setup ---
with st.sidebar:
    st.header("Setup")
    company_name = st.text_input("Target company", value=st.session_state.get("company_name", ""))
    role = st.text_input("Target role", value=st.session_state.get("role", ""))

    st.divider()
    st.markdown("**Try asking:**")
    st.markdown("- *Give me a DSA question*")
    st.markdown("- *Give me an easier one*")
    st.markdown("- *What's their interview process like?*")
    st.markdown("- *What should I brush up on for this role?*")
    st.markdown("- *Ask me a behavioral question*")

    if st.button("🔄 Reset conversation"):
        st.session_state.chat_history = []
        st.session_state.thread_id = os.urandom(4).hex()
        st.rerun()

st.session_state.company_name = company_name
st.session_state.role = role

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "interview-prep-session"

# --- Render chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask for a question, company research, or resume feedback...")

if user_input:
    if not company_name or not role:
        st.warning("Please fill in the target company and role in the sidebar first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    result = app.invoke(
                        {
                            "user_input": user_input,
                            "company_name": company_name,
                            "role": role,
                        },
                        config=config,
                    )
                    answer = result["final_result"]
                except Exception as e:
                    answer = f"Something went wrong: {e}"

            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})