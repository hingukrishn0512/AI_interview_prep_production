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
import os

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


# TODO: update this path to your actual resume PDF location
resume_retriever = build_RAG(r"C:\Users\Yash\Desktop\langGraph\AI_interview_prep_coach\Agent\Hingu_Krishn_Resume_compressed.pdf")


# --- nodes ---

def classifier_node(state: state) -> dict:
    """looks at the user's input and decides which of the 4 branches to route to"""
    user_message = state['user_input']

    prompt = f"""You are a query classifier for an AI interview prep coach.

    Your task is to look at the user's message and decide which category it belongs to.

    Reply with only ONE word, nothing else:
    - "dsa" if they want a coding or DSA practice question
    - "behavioral" if they want a behavioral or HR-style interview question
    - "company_research" if they're asking about the company's interview process, culture, or news
    - "resume_gap" if they're asking what to brush up on based on their resume vs the role

    user_input:
    {user_message}
    """

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    valid_categories = ["dsa", "behavioral", "company_research", "resume_gap"]
    if category not in valid_categories:
        category = "behavioral"

    return {"classifier": category}


def reviwer_node(state: state) -> dict:
    """detects the difficulty level (hard/medium/easy) the user asked for"""
    user_message = state['user_input']

    prompt = f"identify the keywords from the user_input \
            keywords like hard , medium , easy and in return \
            you only give me only one word answer if user entered it \
            hard , easy , medium \
            user_input\
            {user_message}"

    response = llm.invoke(prompt)
    return {"difficulty_level": response.content.strip().lower()}


def array_hasing_node(state: state) -> dict:
    """asking questions about array_hasing_node topic for DSA"""
    difficulty_level = state['difficulty_level']

    prompt = f"""You are an interviewer creating a DSA practice question.

    Generate exactly ONE interview question on the topic of arrays and hashing,
    at a {difficulty_level} difficulty level.

    CRITICAL INSTRUCTIONS:
    1. Output ONLY the question text itself, nothing else.
    2. Do not include the answer, hints, or explanation.
    3. Do not add labels like "Question:" or markdown formatting.
    4. Keep it realistic, the way it would actually be asked in a real interview.

    difficulty_level:
    {difficulty_level}
    """
    response = llm.invoke(prompt)
    return {"candidate_questions": {"arrays_hashing": response.content.strip()}}


def trees_graphs_node(state: state) -> dict:
    """asking questions about trees_graphs_node topic for DSA"""
    difficulty_level = state['difficulty_level']

    prompt = f"""You are an interviewer creating a DSA practice question.

    Generate exactly ONE interview question on the topic of trees and graphs,
    at a {difficulty_level} difficulty level.

    CRITICAL INSTRUCTIONS:
    1. Output ONLY the question text itself, nothing else.
    2. Do not include the answer, hints, or explanation.
    3. Do not add labels like "Question:" or markdown formatting.
    4. Keep it realistic, the way it would actually be asked in a real interview.

    difficulty_level:
    {difficulty_level}
    """
    response = llm.invoke(prompt)
    return {"candidate_questions": {"trees_graphs": response.content.strip()}}


def dp_node(state: state) -> dict:
    """asking questions about dp_node topic for DSA"""
    difficulty_level = state['difficulty_level']

    prompt = f"""You are an interviewer creating a DSA practice question.

    Generate exactly ONE interview question on the topic of dynamic programming,
    at a {difficulty_level} difficulty level.

    CRITICAL INSTRUCTIONS:
    1. Output ONLY the question text itself, nothing else.
    2. Do not include the answer, hints, or explanation.
    3. Do not add labels like "Question:" or markdown formatting.
    4. Keep it realistic, the way it would actually be asked in a real interview.

    difficulty_level:
    {difficulty_level}
    """
    response = llm.invoke(prompt)
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
    final_question = candidates.get(chosen_topic, list(candidates.values())[0])

    return {"final_result": final_question}


def company_reasearch_node(state: state) -> dict:
    """search for a company's interview process, culture, recent news"""
    company_name = state['company_name']
    role = state['role']

    search_query = f"{company_name} {role} interview process culture recent news"
    search_results = search_tool.invoke(search_query)

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
    return {"final_result": response.content.strip()}


def resume_gap_node(state: state) -> dict:
    """identifies the gap in the resumes"""
    role = state['role']
    company_name = state['company_name']
    user_message = state['user_input']

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

    candidate's question:
    {user_message}
    """
    response = llm.invoke(prompt)
    return {"final_result": response.content.strip()}


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
    return {"final_result": response.content.strip()}


def router_function(state: state):
    """reads the category set by classifier_node and decides which branch to go to next"""
    category = state['classifier']

    if category == "dsa":
        return "dsa"
    elif category == "company_research":
        return "company_research"
    elif category == "resume_gap":
        return "resume_gap"
    elif category == "behavioral":
        return "behavioral"

    return "behavioral"

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

    graph.add_edge(START, "classifier_node")

    graph.add_conditional_edges(
        "classifier_node",
        router_function,
        {
            "dsa": "reviwer_node",
            "company_research": "company_reasearch_node",
            "resume_gap": "resume_gap_node",
            "behavioral": "behavioral_node",
        }
    )

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

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)