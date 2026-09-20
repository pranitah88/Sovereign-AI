"""
RAG query API endpoint — migrated from rag_service.py into the unified backend.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.auth.rbac import require_permission
from backend.services.audit import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RagQueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    debug: bool = False


class RagQueryResponse(BaseModel):
    query: str
    context: str
    sources: list
    result_count: int
    refusal: bool = False
    message: str | None = None
    debug_info: dict | None = None


@router.post("/query", response_model=RagQueryResponse)
async def rag_query(
    body: RagQueryRequest,
    user: Annotated[dict, Depends(require_permission("rag_query"))],
):
    """Query the RAG knowledge base with source citations."""
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query text")

    try:
        from backend.services.rag_engine import query_knowledge_base

        result = query_knowledge_base(
            query,
            top_k=body.top_k,
            user_roles=user.get("roles", []),
            debug=body.debug,
        )
    except ImportError as exc:
        logger.error("RAG engine module is not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="RAG engine is not available",
        ) from exc

    audit_log(
        action="rag_query",
        outcome="success",
        user_id=user["id"],
        username=user["username"],
        details={"query": query[:200], "results": result.get("result_count", 0)},
    )

    return RagQueryResponse(**result)
