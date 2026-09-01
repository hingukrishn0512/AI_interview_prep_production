import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from project_raw import create_graph

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

#  this will be the input
class StartSessionRequest(BaseModel):
    company_name: str
    role: str

#  ouptut thread id which is going intot he chat request
class StartSessionResponse(BaseModel):
    thread_id: str

#  now the user_input and thread id will be feeded in the chat request
class ChatRequest(BaseModel):
    thread_id: str
    user_input: str


class ChatResponse(BaseModel):
    final_result: str
    classifier: str


# --- endpoints ---

@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is up. Useful for deployment platforms
       that ping this URL to check if your service is alive."""
    return {"status": "ok"}

#  in post requests we write repsonse classes inside the function there is request

"""payload: StartSessionRequest (in the function signature) tells FastAPI: 
"validate the incoming JSON body against this shape, and give it to me as payload." 
This is the input.
response_model =  StartSessionResponse (in the decorator) tells FastAPI: 
"whatever I return from this function, validate/serialize it against 
this shape before sending it back." This is the output.
"""

@app.post("/session", response_model=StartSessionResponse)  # ← OUTPUT shape (decorator)
def start_session(payload: StartSessionRequest):  # ← INPUT shape (function parameter)
    """Creates a new conversation session for a given company + role.
       Call this ONCE at the start of a user's conversation."""
#   it will take input as per the defination of the class then generating a thread id

    thread_id = str(uuid.uuid4()) # this will generate stread id
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
            "final_result": "",
            "candidate_questions": {},
        },
    )
   # after the updation of the graph function 
#    we get to return the session response which is thread id
    return StartSessionResponse(thread_id=thread_id)

#  response in chat
#  we have to feed thread id
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
    except Exception as e:
        # if thread_id is invalid/unknown, or something inside a node fails,
        # surface a clean HTTP error instead of a raw stack trace
        raise HTTPException(status_code=400, detail=str(e))

    return ChatResponse(
        final_result=result["final_result"],
        classifier=result["classifier"],
    )

#  dont need reload , render will handle it by the PORT
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)