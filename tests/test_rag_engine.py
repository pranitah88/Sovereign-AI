"""
Tests for RAG Engine: text processing, chunking, and ChromaDB retrieval.
"""

from backend.services.rag_engine import (
    clean_text,
    create_chunks,
    create_chunk_id,
    get_collection_stats,
    query_knowledge_base,
)


def test_clean_text():
    dirty = "MRPL  Refinery\x00 \t \n\n\n\nUnits"
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert "  " not in cleaned
    assert "MRPL Refinery\n\nUnits" == cleaned


def test_create_chunks():
    text = "A" * 2500
    chunks = create_chunks(text, chunk_size=1000, overlap=100)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) <= 1000


def test_create_chunk_id():
    chunk_id = create_chunk_id("report.pdf", 1, 3, "Sample text content")
    assert "report_pdf" in chunk_id
    assert "p1" in chunk_id
    assert "c3" in chunk_id


def test_get_collection_stats():
    stats = get_collection_stats()
    assert stats["status"] == "ready"
    assert stats["total_chunks"] > 0
    assert stats["collection_name"] == "mrpl_knowledge_base"


def test_query_knowledge_base():
    result = query_knowledge_base("refinery capacity", top_k=3)
    assert result["query"] == "refinery capacity"
    assert "sources" in result
    assert result["result_count"] >= 1
    assert "context" in result
