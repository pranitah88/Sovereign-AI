"""
MRPL Sovereign AI Workbench — Central RAG Configuration.

Single source of truth for all ingestion, chunking, indexing,
hybrid retrieval (BM25 + ChromaDB), reranking, and threshold parameters.
All values are configurable via environment variables.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
KB_DIR = BASE_DIR / os.getenv("KB_DIR_NAME", "merged_knowledge_base")
CHROMA_DIR = BASE_DIR / "backend" / "chroma_db"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "mrpl_knowledge_base")

# ── Models (Strictly Local / Air-Gapped Inference) ───────────────────────────
_LOCAL_EMBEDDING_DIR = BASE_DIR / "models" / "embeddings" / "all-MiniLM-L6-v2"
_LOCAL_RERANKER_DIR = BASE_DIR / "models" / "reranker" / "ms-marco-MiniLM-L-6-v2"

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    str(_LOCAL_EMBEDDING_DIR) if _LOCAL_EMBEDDING_DIR.exists() else "models/embeddings/all-MiniLM-L6-v2",
)
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    str(_LOCAL_RERANKER_DIR) if _LOCAL_RERANKER_DIR.exists() else "models/reranker/ms-marco-MiniLM-L-6-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() in ("true", "1", "yes")

# ── Chunking Configuration ──────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
CSV_ROWS_PER_CHUNK = int(os.getenv("RAG_CSV_ROWS_PER_CHUNK", "15"))
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))

# ── Retrieval & Candidate Sizing ─────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "20"))
FUSION_TOP_K = int(os.getenv("RAG_FUSION_TOP_K", "20"))
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "15"))
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "5"))

# ── BM25 Parameters ─────────────────────────────────────────────────────────
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# ── Relevance & Grounding Thresholds ────────────────────────────────────────
# Minimum score required for context to be sent to Gemma.
# If nothing passes threshold, a controlled refusal is returned.
RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.20"))
MAX_DISTANCE_THRESHOLD = float(os.getenv("RAG_MAX_DISTANCE", "0.92"))

# ── OCR Fallback Parameters ─────────────────────────────────────────────────
OCR_MIN_NATIVE_CHARS = int(os.getenv("OCR_MIN_NATIVE_CHARS", "80"))
OCR_DPI = int(os.getenv("OCR_DPI", "100"))
