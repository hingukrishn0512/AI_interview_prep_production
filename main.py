import uuid
import logging
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from project_raw import create_graph, RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview_coach.api")

# --- build the graph ONCE, when the API server starts up ---
# (not on every request - that would be very slow and wasteful)
compiled_graph = create_graph()

app = FastAPI(title="AI Interview Prep Coach API")

# CORS lets a frontend running on a different origin (e.g. a React app on
# localhost:3000) actually call this API from the browser. Without this,
# browsers block the request by default for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for local dev only - restrict this in real production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request/response schemas ---
# Pydantic models define exactly what shape of JSON this API expects and returns.
# FastAPI uses these to auto-validate incoming requests and auto-generate docs.

class StartSessionRequest(BaseModel):
    company_name: str
    role: str


class StartSessionResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    thread_id: str
    user_input: str


class ChatResponse(BaseModel):
    final_result: str
    classifier: str


# --- frontend ---
# serves the chat UI at the root URL, and any other files placed in /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# --- endpoints ---

@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is up. Useful for deployment platforms
       that ping this URL to check if your service is alive."""
    return {"status": "ok"}


@app.post("/session", response_model=StartSessionResponse)
def start_session(payload: StartSessionRequest):
    """Creates a new conversation session for a given company + role.
       Call this ONCE at the start of a user's conversation."""

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # update_state seeds the checkpoint with company_name/role WITHOUT running
    # any graph nodes - it's a direct write to the persisted state for this thread_id
    compiled_graph.update_state(
        config,
        {
            "company_name": payload.company_name,
            "role": payload.role,
            "user_input": "",
            "messages": [],
            "classifier": "",
            "difficulty_level": "",
            "topic": "",
            "requested_topics": [],
            "final_result": "",
            "candidate_questions": {},
        },
    )

    return StartSessionResponse(thread_id=thread_id)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """Sends one user message into an existing session and returns the answer."""

    config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        # only "user_input" needs to be passed here - the checkpointer already
        # has company_name/role/etc persisted from the /session call (or from
        # previous /chat calls in this same thread)
        result = compiled_graph.invoke(
            {"user_input": payload.user_input},
            config=config,
        )
    except RateLimitExceeded as e:
        # The LLM provider (Groq) kept throttling us even after retries.
        # 503 tells the frontend/client this is transient and worth retrying
        # shortly, rather than a bad request on their end.
        logger.error("Rate limit exhausted for thread %s: %s", payload.thread_id, e)
        raise HTTPException(
            status_code=503,
            detail="The AI provider is rate-limiting requests right now. Please wait a few seconds and try again.",
        )
    except Exception as e:
        # Log the FULL traceback server-side so the real cause (bad thread_id,
        # a node bug, a provider error, etc.) is actually visible in the
        # terminal/logs instead of just a bare "400 Bad Request" access log line.
        logger.error("chat() failed for thread %s: %s", payload.thread_id, e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

    return ChatResponse(
        final_result=result["final_result"],
        classifier=result["classifier"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)