"""
MRPL Sovereign AI Workbench — Unified FastAPI Application.

Single entry point consolidating all API routes: auth, chat, documents,
RAG, agent, sandbox, models, admin, audit, and system status.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mrpl_sovereign")


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("MRPL Sovereign AI Workbench — Starting up")
    logger.info("=" * 60)

    # Initialize database.
    from backend.database.seed import seed_database
    from backend.database.connection import get_db_path
    seed_database()
    logger.info("Database initialized and seeded.")
    logger.info("ACTIVE AUTHORITATIVE DATABASE PATH: %s", get_db_path().resolve())

    # Log system event.
    try:
        from backend.database.repositories.system_events import log_event
        log_event("startup", "MRPL Sovereign AI Workbench started", severity="info")
    except Exception as exc:
        logger.warning("Could not log startup event: %s", exc)

    # Check sandbox readiness (informational — no build/pull).
    try:
        from backend.services.sandbox import check_sandbox_ready
        sandbox_status = check_sandbox_ready()
        if sandbox_status["ready"]:
            logger.info("Code sandbox: Docker image ready.")
        else:
            logger.warning(
                "Code sandbox NOT ready: %s — %s",
                sandbox_status["reason"],
                sandbox_status.get("detail", ""),
            )
    except Exception as exc:
        logger.warning("Could not check sandbox readiness: %s", exc)

    # Warm up local voice models (ASR / TTS) if enabled
    try:
        from backend.services.voice.voice_service import get_voice_service
        voice_svc = get_voice_service()
        if voice_svc.config.get("voice", {}).get("model_warmup", False):
            warm_res = voice_svc.warmup()
            logger.info("Voice assistant models warmed up: %s", warm_res)
    except Exception as exc:
        logger.warning("Voice assistant warmup non-fatal warning: %s", exc)

    yield


    # Shutdown.
    logger.info("MRPL Sovereign AI Workbench — Shutting down")

    try:
        from backend.database.repositories.system_events import log_event
        log_event("shutdown", "MRPL Sovereign AI Workbench stopped", severity="info")
    except Exception:
        pass

    from backend.database.connection import close_connection
    close_connection()
    logger.info("Database connection closed.")


# ── Application factory ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="MRPL Sovereign AI Workbench",
        description="On-premise agentic AI workbench for MRPL — fully air-gapped, zero cloud dependency.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS (localhost only) ────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",    # Vite dev server
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Mount sub-routers ────────────────────────────────────────────
    from backend.auth.router import router as auth_router
    from backend.api.chat import router as chat_router
    from backend.api.documents import router as documents_router, outputs_router
    from backend.api.rag import router as rag_router
    from backend.api.agent import router as agent_router
    from backend.api.sandbox import router as sandbox_router
    from backend.api.models import router as models_router
    from backend.api.admin import router as admin_router
    from backend.api.audit import router as audit_router
    from backend.api.status import router as status_router
    from backend.api.approvals import router as approvals_router
    from backend.api.voice import router as voice_router

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(outputs_router)
    app.include_router(rag_router)
    app.include_router(agent_router)
    app.include_router(sandbox_router)
    app.include_router(models_router)
    app.include_router(admin_router)
    app.include_router(audit_router)
    app.include_router(status_router)
    app.include_router(approvals_router)
    app.include_router(voice_router)

    # ── Health check (unauthenticated) ───────────────────────────────

    @app.get("/health")
    async def health():
        """Simple health check for monitoring."""
        return {"status": "ok", "service": "mrpl_sovereign_ai"}

    # ── Global exception handler ─────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


# ── Direct execution ─────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
