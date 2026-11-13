# 🎯 AI Prep Coach — Production Multi-Agent Interview Platform

[![Live Application](https://img.shields.io/badge/Live%20App-Vercel%20CDN-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://aiinterviewprepproduction-static.vercel.app)
[![API Server](https://img.shields.io/badge/Backend%20API-FastAPI%20on%20Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://ai-interview-prep-production-lrvm.onrender.com/health)
[![Admin Analytics](https://img.shields.io/badge/Analytics-Live%20Dashboard-white?style=for-the-badge&logo=google-analytics&logoColor=black)](https://ai-interview-prep-production-lrvm.onrender.com/admin/users)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Orchestration](https://img.shields.io/badge/Agentic%20Orchestrator-LangGraph-FF6F00?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)

An enterprise-grade, multi-agent AI interview prep platform built with **LangGraph**, **FastAPI**, **PostgreSQL (`PostgresSaver`)**, **Redis**, and a decoupled **Vercel Edge CDN** frontend. 

Calibrates coding challenges, STAR behavioral questions, company-specific interview loops, and resume gap analysis tailored to a candidate's uploaded resume and target company.

---

## 🌐 Live Deployments

- **Frontend Application**: [https://aiinterviewprepproduction-static.vercel.app](https://aiinterviewprepproduction-static.vercel.app)
- **Backend API**: [https://ai-interview-prep-production-lrvm.onrender.com](https://ai-interview-prep-production-lrvm.onrender.com)
- **Live User Analytics Dashboard**: [https://ai-interview-prep-production-lrvm.onrender.com/admin/users](https://ai-interview-prep-production-lrvm.onrender.com/admin/users)
- **API Health Check**: [https://ai-interview-prep-production-lrvm.onrender.com/health](https://ai-interview-prep-production-lrvm.onrender.com/health)

---

## 🏗️ System Architecture

```
                       ┌────────────────────────────────────────┐
                       │           Candidate Browser            │
                       │   (Pure B&W ChatGPT-Inspired UI)       │
                       └──────────────────┬─────────────────────┘
                                          │
                        HTTPS / TLS       │   Global Edge CDN (<50ms)
                                          ▼
                       ┌────────────────────────────────────────┐
                       │          Vercel Edge Network           │
                       │        (Static SPA & Assets)           │
                       └──────────────────┬─────────────────────┘
                                          │
                               API_BASE   │   REST Calls / JWT Bearer
                                          ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                   Render Production Container (FastAPI Backend)                  │
 │                                                                                 │
 │   ┌────────────────┐      ┌─────────────────────────┐     ┌─────────────────┐   │
 │   │ Authentication │      │ Redis Rate Limiting     │     │ Token & Latency │   │
 │   │ (JWT / Bcrypt) │      │ (Token-Bucket Algorithm)│     │ Telemetry Logger│   │
 │   └───────┬────────┘      └────────────┬────────────┘     └────────┬────────┘   │
 │           │                            │                           │            │
 │           ▼                            ▼                           ▼            │
 │  ┌───────────────────────────────────────────────────────────────────────────┐  │
 │  │                     LangGraph State Machine Orchestration                 │  │
 │  │                                                                           │  │
 │  │                              ┌───────────────┐                            │  │
 │  │                              │  START NODE   │                            │  │
 │  │                              └───────┬───────┘                            │  │
 │  │                                      │                                    │  │
 │  │                                      ▼                                    │  │
 │  │                           ┌─────────────────────┐                         │  │
 │  │                           │   Classifier Node   │                         │  │
 │  │                           │  (Intent Analysis)  │                         │  │
 │  │                           └──────────┬──────────┘                         │  │
 │  │                                      │                                    │  │
 │  │       ┌──────────────┬───────────────┼───────────────┬─────────────┐      │  │
 │  │       │              │               │               │             │      │  │
 │  │       ▼              ▼               ▼               ▼             ▼      │  │
 │  │  ┌─────────┐   ┌────────────┐  ┌───────────┐  ┌─────────────┐ ┌─────────┐ │  │
 │  │  │   DSA   │   │ Behavioral │  │  Company  │  │ Resume Gap  │ │ General │ │  │
 │  │  │ Reviewer│   │ STAR Coach │  │ Research  │  │  Analysis   │ │  Coach  │ │  │
 │  │  │  Node   │   └─────┬──────┘  │ (Tavily)  │  │(RAG + FAISS)│ └────┬────┘ │  │
 │  │  └────┬────┘         │         └─────┬─────┘  └──────┬──────┘      │      │  │
 │  │       │ (Parallel)   │               │               │             │      │  │
 │  │   ┌───┴───┐          │               │               │             │      │  │
 │  │   │Arrays/│          │               │               │             │      │  │
 │  │   │Trees/ │          │               │               │             │      │  │
 │  │   │  DP   │          │               │               │             │      │  │
 │  │   └───┬───┘          │               │               │             │      │  │
 │  │       ▼              │               │               │             │      │  │
 │  │  ┌─────────┐         │               │               │             │      │  │
 │  │  │ Picker  │         │               │               │             │      │  │
 │  │  │  Node   │         │               │               │             │      │  │
 │  │  └────┬────┘         │               │               │             │      │  │
 │  │       │              │               │               │             │      │  │
 │  │       └──────────────┴───────────────┼───────────────┴─────────────┘      │  │
 │  │                                      ▼                                    │  │
 │  │                               ┌─────────────┐                             │  │
 │  │                               │  END NODE   │                             │  │
 │  │                               └─────────────┘                             │  │
 │  └──────────────────────────────────────┬────────────────────────────────────┘  │
 └─────────────────────────────────────────┼───────────────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
        ┌───────────────────────────────┐     ┌───────────────────────────────┐
        │   PostgreSQL (Cloud Database) │     │      Redis (Key-Value Cache)  │
        │ - PostgresSaver Checkpointer  │     │ - Sliding-Window Rate Limits  │
        │ - User Accounts & Auth Hashes │     │ - Cost Protection Sentinel    │
        │ - Messages, Tokens & Latency  │     │ - Fast Ephemeral States       │
        └───────────────────────────────┘     └───────────────────────────────┘
```

---

## 🚀 The Upgrade Journey: From Local Script to Production Product

Most AI engineering tutorials build toy prototypes that only run in a local notebook or stream-lit wrapper on `localhost:8501`. 

This repository documents the transition of an interview coach into a hardened, enterprise-ready cloud product:

| Capability | Initial Prototype (Laptop Demo) | Production Platform (Current Architecture) |
| :--- | :--- | :--- |
| **Hosting & Topology** | Single monolithic Python script | **Decoupled**: Vercel Edge CDN (Frontend) + Render Cloud (FastAPI Backend) |
| **Persistence** | In-memory / Ephemeral state | **PostgreSQL `PostgresSaver`** multi-turn checkpointing (7-day state retention) |
| **Security & Auth** | None (public / single-user) | **JWT Bearer Auth**, Bcrypt hashing, isolated session tenancies |
| **Cost & Abuse Protection** | None (vulnerable to API bill drain) | **Redis token-bucket rate limiter** (`chat_per_day`, `session_per_hour`) |
| **Telemetry & Observability** | Raw print statements | Structured JSON logging with **`tokens_in`**, **`tokens_out`**, and **`latency_ms`** per turn |
| **Router Reliability** | Blind LLM prompt | **20-Question Golden Evaluation Suite** (`eval_router.py`) asserting $\ge 90\%$ accuracy |
| **Admin Analytics** | Direct manual SQL shell queries | **Live Server-Rendered HTML Dashboard** (`/admin/users`) with real-time stats |

---

## 🧗 Where We Got Stuck & How We Solved It (Engineering Post-Mortem)

Taking an agentic LLM application into production exposes non-obvious failure modes that never appear on a local machine. Here are the core challenges encountered during deployment and how they were solved:

### 1. Database Networking & Cloud Migration
- **The Failure**: When deploying to Render, the backend crashed with `socket.gaierror: [Errno -2] Name or service not known` and `ModuleNotFoundError: No module named alembic`.
- **Root Cause**: The local setup relied on Docker Compose internal hostnames (`@db:5432` and `@redis:6379`) that do not resolve in isolated cloud container networks. Additionally, untracked Alembic migration folders were missing from the production branch.
- **The Fix**: 
  - Separated internal and external database connection strings.
  - Configured `DATABASE_URL` with explicit SSL query parameters (`?ssl=require`) for asyncpg.
  - Tracked migration folders and configured Render with Python 3.12 (`PYTHON_VERSION=3.12.8`).

### 2. Frontend Latency & UI Render Degradation
- **The Failure**: Serving single-page static files directly from Render's Python web container resulted in slow initial page loads, high TTFB (Time to First Byte), and noticeable frame lag when rendering complex chat streams.
- **Root Cause**: Python application servers are not optimized for static file edge caching. Furthermore, GPU-intensive CSS properties (`backdrop-filter: blur()`) caused performance drops on lower-spec candidate devices.
- **The Fix**:
  - Decoupled the frontend and deployed static assets directly to **Vercel's Edge CDN** (<50ms worldwide asset delivery).
  - Connected the client to the backend via a configurable `const API_BASE`.
  - Re-engineered the UI into a pure **ChatGPT-inspired black & white monochrome theme** with zero blur filters and zero layout shift.

### 3. Silent Session Checkpoint Wiping
- **The Failure**: When users clicked "New Chat" or "Log out" in the UI, past conversation checkpoints were vanishing from PostgreSQL.
- **Root Cause**: Client-side navigation handlers were indiscriminately triggering backend teardown endpoints (`endServerSession()`), deleting active thread checkpoints.
- **The Fix**:
  - Decoupled UI state resets from server data deletion.
  - Increased session persistence window to **7 days** (`SESSION_MAX_AGE_SECONDS = 604800`).
  - Added dedicated, explicit user actions: `DELETE /sessions/{thread_id}` (purge chat) and `DELETE /session/{thread_id}` (read-only mode removing resume bytes).

### 4. Silent Multi-Agent Misrouting
- **The Failure**: In a multi-agent system, the classifier router is the highest-risk single point of failure. When it misclassifies, the user receives a plausible answer from the wrong agent, HTTP response codes remain `200 OK`, and rate limits/latency metrics stay green.
- **Root Cause**: Intent boundaries between broad requests (e.g. *"Give me an HR question"*) and targeted requests (e.g. *"Give me a behavioral question about conflict resolution"*) are subtle and prone to prompt drift.
- **The Fix**:
  - Formalized strict disambiguation rules in the classifier prompt.
  - Authored an automated regression evaluation suite: [`Agent/eval_router.py`](file:///C:/Users/Yash/Desktop/langGraph/AI_interview_prep_coach/Agent/eval_router.py).
  - Evaluates **20 ground-truth interview queries** across all 5 branches on every build, rejecting deploys if accuracy drops below $90\%$.

### 5. Quadratic Context Window Explosion & Cost Spikes
- **The Failure**: As users engaged in long mock interview loops, LLM inference latency increased and token costs scaled quadratically.
- **Root Cause**: History state was being re-injected into intermediate nodes without step-level token accounting, obscuring cost-per-turn metrics.
- **The Fix**:
  - Implemented LangChain usage metadata callbacks in `main.py`.
  - Extracted and tracked `tokens_in` and `tokens_out` per step.
  - Recorded token consumption and latency into structured JSON logs and the PostgreSQL `messages` table for per-user auditing.

### 6. Cross-Platform Database Telemetry vs. Live Admin Dashboard
- **The Failure**: Monitoring active users required running Linux bash commands (`PGPASSWORD=... psql`), which fail on Windows PowerShell (`CommandNotFoundException`).
- **The Fix**:
  - Engineered a native FastAPI analytics endpoint: [`GET /admin/users`](file:///C:/Users/Yash/Desktop/langGraph/AI_interview_prep_coach/Agent/main.py#L222).
  - Serves an **HTML Analytics Dashboard** with summary cards (Total Users, Total Sessions, Total Messages, Active Candidates), per-user session drill-downs, and target company statistics.
  - Supports programmatic JSON exports via `?format=json`.

---

## ⚡ Core Agent Workflows

The LangGraph engine directs incoming candidate input through 5 specialized nodes:

1. **`dsa` (Data Structures & Algorithms)**:
   - Evaluates target company difficulty level (Easy/Medium/Hard).
   - Generates candidate questions across 3 parallel tracks: *Arrays & Hashing*, *Trees & Graphs*, and *Dynamic Programming*.
   - A dedicated `picker_node` selects the optimal problem matching the candidate's target company profile and prevents repeating past questions.
2. **`behavioral` (STAR Method Coaching)**:
   - Conducts behavioral mock interviews focusing on leadership, conflict management, cross-functional collaboration, and overcoming failure.
   - Requires candidates to answer in the **Situation-Task-Action-Result (STAR)** format.
3. **`company_research` (Live Company Intel)**:
   - Grounded via **Tavily Web Search** to fetch real-time interview formats, hiring rounds, and recent engineering news.
4. **`resume_gap` (Resume-Grounded Gap Analysis)**:
   - Employs a vector-similarity RAG pipeline (over the candidate's parsed PDF) to spot background gaps relative to the target role.
5. **`general_chat` (Interview Strategy & Clarifications)**:
   - Handles underspecified prompts, clarifications, interview study resources, and general guidance.

---

## 📊 Live Admin Analytics Dashboard

Access live metrics directly in your browser:
👉 **`https://ai-interview-prep-production-lrvm.onrender.com/admin/users`**

- **Metric Cards**: Total Registered Users, Active Sessions, Total Q&A Messages, Active User Conversion.
- **Per-User Breakdown**: Email, Registration Date, Sessions Started, Message Volume, Last Active Timestamp, Agent Breakdown (DSA vs. Behavioral vs. Company Intel).
- **Target Calibration**: Real-time company and role targets (e.g. `Software Engineer @ Google`, `ML Engineer @ Meta`).
- **Export**: Append `?format=json` for machine-readable JSON telemetry.

---

## 🧪 Automated Routing Regression Suite

To verify that the classifier correctly routes candidate prompts without regressions, run the golden evaluation suite:

```bash
python eval_router.py
```

### Output Example:
```text
======================================================================
🚀 RUNNING MULTI-AGENT ROUTER EVALUATION (20 Ground-Truth Samples)
======================================================================
[01/20] ✅ PASS | 412ms | Query: "Give me a hard dynamic programming question on the knapsack..."
[02/20] ✅ PASS | 385ms | Query: "Can you give me an array problem with two pointers or sliding..."
[03/20] ✅ PASS | 390ms | Query: "Ask me a tree traversal question using recursion..."
...
[20/20] ✅ PASS | 370ms | Query: "What are some good resources or books to learn system design?..."

======================================================================
EVALUATION SUMMARY: 20/20 Passed (100.0% Accuracy)
Average Classifier Latency: 395 ms
======================================================================
✅ ROUTER EVALUATION PASSED. System is safe to deploy.
```

---

## 🛠️ Local Development & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/hingukrishn0512/AI_interview_prep_production.git
cd AI_interview_prep_production
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
# LLM & Search Providers
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token

# PostgreSQL (SQLAlchemy Async + PostgresSaver Sync)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/coach
CHECKPOINT_URL=postgresql://user:password@localhost:5432/coach

# Redis & Authentication
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your_jwt_secret_key
```

### 3. Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run Migrations & Start Server
```bash
alembic upgrade head
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000` to interact with the local application.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
