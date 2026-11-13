import os
import uuid
import time
import asyncio
import logging
import tempfile
import threading
import traceback

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog


from auth import router as auth_router, get_current_user
from config import settings
from db.models import User, ChatSession, Message
from db.session import get_db
from logging_setup import setup_logging
from ratelimit import chat_per_day, chat_per_minute, session_per_hour


setup_logging()
logger = logging.getLogger("interview_coach.api")
metrics_log = structlog.get_logger("interview_coach.metrics")

MAX_RESUME_BYTES = 5 * 1024 * 1024  # 5 MB
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # sessions inactive for 7 days get cleaned up
PURGE_EVERY_SECONDS = 60 * 60               # run the cleanup query at most once an hour
_last_purge = 0.0

# ----------------------------------------------------
# The heavy part (importing project_raw = LangChain/LangGraph, then building
# the graph) runs in a BACKGROUND thread. The web server starts answering
# requests within a second or two, so the page and /health work immediately
# while the AI engine is still loading.
# ----------------------------------------------------
pr = None               # the project_raw module, once imported
compiled_graph = None   # the compiled LangGraph graph
load_error = None



def _load_in_background():
    global pr, compiled_graph, load_error
    try:
        import project_raw

        # CHECKPOINT_URL (plain postgresql://) wins if set in .env; otherwise it
        # is derived from DATABASE_URL, so there is only one URL to keep in sync.
        checkpoint_url = os.getenv("CHECKPOINT_URL") or settings.sync_database_url

        graph = project_raw.create_graph(checkpoint_url)

        compiled_graph = graph
        pr = project_raw
        logger.info("Graph is ready")

    except Exception as e:
        load_error = e
        logger.error("Graph failed to build: %s", e)
        traceback.print_exc()



threading.Thread(target=_load_in_background, daemon=True).start()


def is_ready():
    return pr is not None and compiled_graph is not None


def require_ready():
    if load_error is not None:
        raise HTTPException(
            status_code=500,
            detail="The AI engine failed to start. Please try again later.",
        )
    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="The server is still warming up. Please try again in a few seconds.",
        )


app = FastAPI(title="AI Interview Prep Coach API")
app.include_router(auth_router)
# The frontend is served by this same app, so CORS is only needed if you host the
# UI somewhere else (the wake-up page needs it to check /health). Credentials are
# off because "*" origins + credentials is not a valid combination.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# One request_id per request, attached to every log line made while handling it
# (including the ones inside the graph's worker threads), and returned to the
# client in the X-Request-ID header so a bug report can be traced in the logs.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = (request.headers.get("X-Request-ID") or uuid.uuid4().hex)[:32]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        raise
    response.headers["X-Request-ID"] = request_id
    path = request.url.path
    if path != "/health" and not path.startswith("/static"):
        metrics_log.info(
            "request",
            method=request.method,
            path=path,
            status=response.status_code,
            duration_ms=int((time.perf_counter() - started) * 1000),
            user_id=getattr(request.state, "user_id", None),
        )
    return response


# --- request/response schemas ---

class StartSessionResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    thread_id: str
    user_input: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    final_result: str
    classifier: str


# --- helpers ---

def drop_thread(thread_id: str):
    """Deletes the persisted graph state, including resume text, for a session.

    The SQLAlchemy Message rows are intentionally not deleted, so History can
    still display the old conversation after resume data is removed.
    """
    if not is_ready():
        return

    try:
        if pr._saver is not None:
            pr._saver.delete_thread(thread_id)
    except Exception as e:
        logger.warning("Couldn't delete checkpoints for %s: %s", thread_id, e)


def cleanup_old_sessions():
    """Best-effort cleanup, run at most once an hour. A cleanup-query problem
    must not block new sessions."""
    global _last_purge
    if not is_ready():
        return
    if time.time() - _last_purge < PURGE_EVERY_SECONDS:
        return
    _last_purge = time.time()

    try:
        purged_threads = pr.purge_inactive_threads(SESSION_MAX_AGE_SECONDS)
        for thread_id in purged_threads:
            logger.info("Cleaned up expired session %s", thread_id)
    except Exception as e:
        # The checkpoint timestamp query depends on the installed LangGraph
        # checkpoint schema. Keep session creation working if that query needs
        # adjustment; verify the query separately before relying on cleanup.
        logger.warning("Automatic checkpoint cleanup skipped: %s", e)


async def get_owned_session(db: AsyncSession, thread_id: str, user: User) -> ChatSession:
    """Returns the session only if it belongs to this user, otherwise 404.
    Using 404 (not 403) means users can't probe for other people's session IDs."""
    try:
        sid = uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found.")
    row = await db.scalar(
        select(ChatSession).where(ChatSession.id == sid, ChatSession.user_id == user.id)
    )
    if not row:
        raise HTTPException(status_code=404, detail="Session not found.")
    return row


# --- frontend ---
# serves the chat UI at the root URL, and any other files placed in /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})


# --- endpoints ---

@app.get("/health")
def health_check():
    """Confirms the server is up. 'ready' turns true once the AI engine has loaded."""
    return {"status": "ok", "ready": is_ready()}


@app.get("/admin/users")
async def get_admin_users(db: AsyncSession = Depends(get_db)):
    """Returns live registration count and user list."""
    res = await db.execute(select(User.email, User.created_at).order_by(User.created_at.desc()))
    users = res.all()
    session_count = await db.scalar(select(func.count(ChatSession.id)))
    message_count = await db.scalar(select(func.count(Message.id)))
    return {
        "total_users": len(users),
        "total_sessions": session_count,
        "total_messages": message_count,
        "users": [
            {
                "email": u[0],
                "joined": u[1].strftime("%Y-%m-%d %H:%M:%S") if u[1] else None,
            }
            for u in users
        ],
    }


@app.post("/session", response_model=StartSessionResponse)
async def start_session(
    company_name: str = Form(...),
    role: str = Form(...),
    resume: UploadFile = File(...),
    user: User = Depends(session_per_hour),
    db: AsyncSession = Depends(get_db),
):
    """Creates a new conversation session for the logged-in user.
       Call this ONCE at the start of a conversation."""

    require_ready()

    company_name = company_name.strip()
    role = role.strip()
    if not company_name or not role:
        raise HTTPException(status_code=400, detail="Company and role are required.")

    await asyncio.to_thread(cleanup_old_sessions)

    # Read one byte past the limit, so oversized files are rejected without
    # loading the whole thing into memory.
    data = await resume.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="Resume is too large (max 5 MB).")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    thread_id = str(uuid.uuid4())

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Extract the PDF text. The PDF file itself is temporary; the extracted
        # text is saved in the LangGraph PostgreSQL checkpoint below.
        resume_text = await asyncio.to_thread(pr.extract_resume_text, tmp_path)
    except ValueError as e:
        logger.error("Resume has no readable text: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Couldn't read text from that PDF. Make sure it isn't a scanned image.",
        )
    except Exception as e:
        logger.error("Resume processing failed: %s", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Couldn't process your resume. Please try again.",
        )
    finally:
        # The uploaded PDF itself is never kept on disk.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    try:
        # Seed the new thread's initial state without running any graph nodes.
        config = {"configurable": {"thread_id": thread_id}}
        await asyncio.to_thread(
            compiled_graph.update_state,
            config,
            {
                "company_name": company_name,
                "role": role,
                "user_input": "",
                "messages": [],
                "classifier": "",
                "difficulty_level": "",
                "topic": "",
                "requested_topics": [],
                "final_result": "",
                "candidate_questions": {},
                "resume_text": resume_text,
            },
        )
    except Exception as e:
        logger.error("Couldn't create session state: %s", e)
        await asyncio.to_thread(drop_thread, thread_id)
        raise HTTPException(status_code=500, detail="Couldn't start the session. Please try again.")

    # Only create the DB row after everything else succeeded, so failures leave no orphan rows
    try:
        db.add(
            ChatSession(
                id=uuid.UUID(thread_id),
                user_id=user.id,
                title=f"{role} @ {company_name}"[:200],
            )
        )
        await db.commit()
    except Exception as e:
        logger.error("Couldn't save session row: %s", e)
        await db.rollback()
        await asyncio.to_thread(drop_thread, thread_id)
        raise HTTPException(status_code=500, detail="Couldn't start the session. Please try again.")

    return StartSessionResponse(thread_id=thread_id)


@app.delete("/session/{thread_id}")
async def end_session(
    thread_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes persisted resume/graph state. The separate DB chat history is kept."""
    await get_owned_session(db, thread_id, user)
    require_ready()
    await asyncio.to_thread(drop_thread, thread_id)
    return {"status": "ok"}


@app.delete("/sessions/{thread_id}")
async def delete_session(
    thread_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes the chat session, all of its messages, and any checkpoint/resume data."""
    session_row = await get_owned_session(db, thread_id, user)
    require_ready()
    session_id = session_row.id
    await asyncio.to_thread(drop_thread, thread_id)
    await db.execute(delete(Message).where(Message.session_id == session_id))
    await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
    await db.commit()
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(chat_per_day)])
async def chat(
    payload: ChatRequest,
    user: User = Depends(chat_per_minute),
    db: AsyncSession = Depends(get_db),
):
    """Sends one user message into one of the caller's sessions and returns the answer."""

    require_ready()

    user_text = payload.user_input.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_row = await get_owned_session(db, payload.thread_id, user)
    session_id = session_row.id
    # Release the DB connection now, so it isn't held open during the slow LLM call
    await db.rollback()

    # Resume text is stored in the PostgreSQL checkpoint, so it survives API
    # restarts. If the checkpoint was deliberately deleted, Continue is unavailable.
    config = {"configurable": {"thread_id": payload.thread_id}}
    try:
        snapshot = await asyncio.to_thread(compiled_graph.get_state, config)
    except Exception as e:
        logger.error("Couldn't load session state for %s: %s", payload.thread_id, e)
        raise HTTPException(
            status_code=500,
            detail="Couldn't load this session. Please try again.",
        )
    if not snapshot or not snapshot.values or not snapshot.values.get("resume_text"):
        raise HTTPException(
            status_code=404,
            detail="Resume data is unavailable for this session. Start a new session and upload your resume again.",
        )

    # Collects token usage from every LLM call the graph makes for this request.
    # Imported here because langchain is loaded in the background at startup.
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    usage_cb = UsageMetadataCallbackHandler()
    run_config = {**config, "callbacks": [usage_cb]}

    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            compiled_graph.invoke, {"user_input": user_text}, config=run_config
        )
    except pr.RateLimitExceeded as e:
        # 503 tells the frontend this is transient and worth retrying shortly.
        logger.error("Rate limit exhausted for thread %s: %s", payload.thread_id, e)
        raise HTTPException(
            status_code=503,
            detail="The AI provider is rate-limiting requests right now. Please wait a few seconds and try again.",
        )
    except Exception as e:
        # Log the FULL traceback server-side so the real cause is visible in the logs.
        logger.error("chat() failed for thread %s: %s", payload.thread_id, e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer. Please try again.",
        )
    latency_ms = int((time.perf_counter() - started) * 1000)

    usage = usage_cb.usage_metadata or {}
    tokens_in = sum(u.get("input_tokens", 0) for u in usage.values()) if usage else None
    tokens_out = sum(u.get("output_tokens", 0) for u in usage.values()) if usage else None
    metrics_log.info(
        "chat_completed",
        thread_id=payload.thread_id,
        agent=result["classifier"],
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    # Saving history must never break a chat the user already got an answer for
    try:
        db.add_all([
            Message(session_id=session_id, role="user", content=user_text),
            Message(
                session_id=session_id,
                role="assistant",
                content=result["final_result"],
                agent_used=result["classifier"],
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            ),
        ])
        await db.commit()
    except Exception as e:
        logger.error("Couldn't save messages for %s: %s", payload.thread_id, e)
        await db.rollback()

    return ChatResponse(
        final_result=result["final_result"],
        classifier=result["classifier"],
    )


# --- history endpoints (base for the future progress dashboard) ---

@app.get("/sessions")
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
    )
    return [{"id": str(s.id), "title": s.title, "created_at": s.created_at} for s in rows]


@app.get("/sessions/{thread_id}/status")
async def session_status(
    thread_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Reports whether this owned session still has resume/checkpoint data."""
    await get_owned_session(db, thread_id, user)
    require_ready()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await asyncio.to_thread(compiled_graph.get_state, config)
        can_continue = bool(snapshot and snapshot.values and snapshot.values.get("resume_text"))
    except Exception as e:
        logger.warning("Couldn't inspect checkpoint for %s: %s", thread_id, e)
        can_continue = False
    return {"can_continue": can_continue}


@app.get("/sessions/{thread_id}/messages")
async def session_messages(
    thread_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await get_owned_session(db, thread_id, user)
    rows = await db.scalars(
        select(Message).where(Message.session_id == s.id).order_by(Message.created_at)
    )
    return [
        {"role": m.role, "content": m.content, "agent": m.agent_used, "latency_ms": m.latency_ms}
        for m in rows
    ]


if __name__ == "__main__":
    import uvicorn
    # Keep a single worker so this process owns one shared graph/checkpointer pool.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)