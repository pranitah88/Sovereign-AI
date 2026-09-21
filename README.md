# MRPL Sovereign AI Workbench

An on-premise, air-gapped AI workbench built for industrial operations at Mangalore Refinery and Petrochemicals Limited (MRPL). All inference, data storage, and code execution happen locally — nothing leaves the premises.

## Problem

Industrial facilities handle sensitive operational data — engineering specifications, equipment manuals, process parameters, and safety procedures. Using cloud-based AI services introduces unacceptable risks: data exfiltration, vendor lock-in, latency over unreliable plant networks, and compliance violations.

The Sovereign AI Workbench eliminates cloud dependency entirely. It runs a complete AI assistant stack on a single local machine, giving engineers secure access to intelligent document retrieval, code execution, vision analysis, and voice interaction without any internet connection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                    │
│           Chat · Dashboard · Documents · Admin · Audit      │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend (Uvicorn)                  │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Auth (RBAC) │  │ Agent Graph  │  │ Security Guards    │  │
│  │ JWT/Session │  │ (LangGraph)  │  │ Scope · Temporal · │  │
│  │ 4 Roles     │  │ Multi-step   │  │ Confidence · Net   │  │
│  └─────────────┘  └──────┬───────┘  └────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼────────────────────────────────┐ │
│  │              Hybrid RAG Engine                         │ │
│  │   ChromaDB (vectors) + BM25 (lexical) + Reranking      │ │
│  │   Pre-LLM clearance filtering · Citation validation    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐   │
│  │ Sandbox  │  │ Voice     │  │ Vision   │  │ DocGen    │   │
│  │ (Docker) │  │ ASR + TTS │  │ OCR/VL   │  │ DOCX/XLSX │   │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ localhost:11434
              ┌────────────▼────────────────┐
              │     Ollama (Local LLMs)     │
              │  Gemma 3 4B · Qwen 2.5      │
              │  Coder 3B · Qwen 2.5 VL 3B  │
              └─────────────────────────────┘
```

## Key Features

- **Conversational AI Agent**: Multi-step agentic reasoning powered by LangGraph with tool use, planning, and iterative refinement
- **Hybrid RAG Pipeline**: ChromaDB vector search + BM25 lexical retrieval + cross-encoder reranking with source citations
- **Isolated Code Execution**: Docker sandbox with `network_mode="none"`, read-only root filesystem, 256 MB RAM limit, non-root user, and 30-second timeout
- **Voice Assistant**: Real-time speech-to-text (Whisper ASR) and text-to-speech with voice activity detection, language detection, and domain vocabulary support
- **Vision Analysis**: Engineering drawing and P&ID analysis using multimodal models (Gemma 3, Qwen 2.5 VL) with PaddleOCR text extraction
- **Document Generation**: Automated DOCX, XLSX, and PPTX report generation with AI-generated watermarking
- **Role-Based Access Control**: Four roles (Admin, Engineer, Reviewer, Viewer) with data classification clearance (Public, Internal, Confidential, Highly Confidential)
- **Human-in-the-Loop Approvals**: High-risk actions require explicit reviewer/admin approval before execution
- **Audit Trail**: Immutable SQLite logging of all queries, tool executions, authentication events, and approval decisions
- **Temporal Intent Detection**: Distinguishes between current operational queries and historical data requests
- **Multilingual Support**: Query processing and response generation across multiple languages
- **n8n Workflow Integration**: Document ingestion and scheduled reindexing workflows via local n8n instance

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, WebGL (MeshGradient) |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| Agent Framework | LangGraph, LangChain |
| LLM Inference | Ollama (local, air-gapped) |
| Models | Gemma 3 4B, Qwen 2.5 Coder 3B, Qwen 2.5 VL 3B |
| Vector Database | ChromaDB (local SQLite storage) |
| Embeddings | sentence-transformers (local) |
| OCR | PaddleOCR, PaddlePaddle |
| Code Sandbox | Docker (isolated container) |
| Database | SQLite (WAL mode) |
| Auth | Argon2 password hashing, session-based |
| Workflow Automation | n8n (Docker Compose) |
| Voice | Whisper ASR, local TTS |

## Security Design

The workbench operates under a **zero-trust architecture**:

1. **Air-gapped inference** — All LLM calls go to `localhost:11434` (Ollama). No external API keys required or accepted.
2. **Pre-LLM clearance filtering** — Retrieved documents exceeding the user's clearance level are dropped before the model sees them.
3. **Fail-closed sandbox** — If Docker is unreachable or the sandbox image is missing, code execution fails immediately. There is no host fallback.
4. **Approved model registry** — Only models explicitly marked `approved: true` in `config/model_registry.yaml` can be used.
5. **Prompt injection defense** — Retrieved passages are wrapped in untrusted content delimiters; regex scanners detect override attempts.
6. **Network seal verification** — Runtime checks confirm no external AI endpoints are reachable.

See [SECURITY.md](SECURITY.md) for the complete security architecture, threat model, and demonstration procedures.

## Repository Structure

```
├── backend/
│   ├── agent/              # LangGraph agent (graph, state, tools, prompts)
│   ├── api/                # FastAPI route modules (chat, rag, voice, sandbox, admin, audit)
│   ├── auth/               # Authentication, RBAC, session management
│   ├── database/           # SQLite schema, seed data, repository pattern
│   ├── services/           # Core services
│   │   ├── voice/          # ASR, TTS, VAD, conversation state, language detection
│   │   ├── rag_engine.py   # Hybrid RAG (ChromaDB + BM25 + reranking)
│   │   ├── sandbox.py      # Docker sandbox execution
│   │   ├── security_guards.py
│   │   ├── temporal_guard.py
│   │   ├── scope_guard.py
│   │   ├── confidence_gate.py
│   │   └── ...
│   ├── app.py              # Application factory and lifespan
│   ├── rag_config.py       # RAG pipeline configuration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/          # Chat, Dashboard, Documents, Admin, Audit, Login
│   │   ├── components/     # HeroSection, VoiceOrb, KnowledgeExplorer, CodingResponseCard
│   │   ├── api/            # Backend API client
│   │   └── lib/            # Utilities
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── config/
│   ├── model_registry.yaml # Approved local models and task routing
│   ├── scope_rules.yaml    # Query scope enforcement rules
│   └── voice.yaml          # Voice assistant configuration
├── n8n/
│   ├── docker-compose.yml  # Local n8n orchestration
│   └── workflows/          # Document ingestion and reindexing workflows
├── tests/                  # Automated test suite (261+ tests)
├── scripts/                # Verification and utility scripts
├── SECURITY.md             # Security architecture documentation
└── .gitignore
```

## Local Setup

### Prerequisites

- Python 3.13+
- Node.js 18+
- [Ollama](https://ollama.com/) installed locally
- Docker Desktop (for code sandbox)

### 1. Clone and set up the backend

```bash
git clone https://github.com/pranitah88/Sovereign-AI.git
cd Sovereign-AI

python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r backend/requirements.txt
```

### 2. Pull the required models

```bash
ollama pull gemma3:4b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-vl:3b
```

### 3. Build the sandbox image

```bash
docker build -t mrpl-sandbox:latest -f backend/services/Dockerfile.sandbox .
```

### 4. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 5. Start the application

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — Backend
.\venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Terminal 3 — Frontend
cd frontend
npm run dev
```

The application will be available at `http://localhost:5173`.

### 6. Run the test suite

```bash
.\venv\Scripts\pytest -q
```

## License

This project was developed as part of the Smart India Hackathon (SIH) initiative. See individual model licenses in `config/model_registry.yaml`.
