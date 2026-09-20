"""
Local In-Memory BM25 Lexical Search Engine.

Fully offline, pure Python, zero cloud dependencies.
Designed specifically for industrial/technical documents, preserving:
- Exact part/unit names (PFCCU, DCU, DHDT, CCR, CDU, VDU, SPM, APMC)
- Filenames and extensions (.csv, .pdf, .txt, .docx, .json)
- Numbers, metrics, word counts, byte sizes (e.g. 9000, 847, 1256307)
- Compound names and snake_case / camelCase identifiers.
"""

import logging
import math
import re
from collections import Counter
from typing import Any

from backend.rag_config import BM25_B, BM25_K1

logger = logging.getLogger(__name__)

# Basic stop words to eliminate noise while preserving domain terms
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "not", "this", "that", "it",
    "from", "into", "through", "during", "before", "after", "above", "below",
}


def tokenize_technical_text(text: str) -> list[str]:
    """
    Tokenize text preserving technical acronyms, filenames, numbers, and codes.
    Generates both composite tokens and constituent parts.
    """
    if not text:
        return []

    tokens: list[str] = []
    text_lower = text.lower()

    # 1. Capture exact filenames with extensions (e.g. coverage_report_paddle.csv)
    filenames = re.findall(r'[\w\-]+\.(?:txt|pdf|csv|docx|json|xlsx|pptx)', text_lower)
    tokens.extend(filenames)

    # 2. Extract alphanumeric words and technical tokens including underscores
    raw_words = re.findall(r'[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*', text_lower)

    for w in raw_words:
        if not w or w in STOP_WORDS:
            continue
        tokens.append(w)
        # If token has underscore or hyphen, split into components as well
        if "_" in w or "-" in w:
            parts = [p for p in re.split(r'[_\-]+', w) if p and p not in STOP_WORDS]
            tokens.extend(parts)

    return tokens


class BM25Index:
    """
    Okapi BM25 in-memory index built from ChromaDB chunk texts.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        # Inverted index: term -> list of (doc_idx, term_frequency)
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        # Precomputed IDF cache
        self.idf_cache: dict[str, float] = {}

    def fit(self, chunk_ids: list[str], documents: list[str]) -> "BM25Index":
        """Build the inverted index from document collection."""
        self.corpus_ids = list(chunk_ids)
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.inverted_index = {}
        self.idf_cache = {}

        if self.doc_count == 0:
            self.avg_doc_len = 0.0
            return self

        total_length = 0
        doc_frequencies: dict[str, int] = Counter()

        for doc_idx, doc_text in enumerate(documents):
            tokens = tokenize_technical_text(doc_text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            counts = Counter(tokens)
            for term, freq in counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))
                doc_frequencies[term] += 1

        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 0.0

        # Precalculate Okapi BM25 IDF: ln( (N - n + 0.5)/(n + 0.5) + 1 )
        for term, n_q in doc_frequencies.items():
            idf = math.log(((self.doc_count - n_q + 0.5) / (n_q + 0.5)) + 1.0)
            self.idf_cache[term] = max(0.01, idf)

        logger.info(
            "BM25 index constructed with %d documents, %d unique terms, avgdl=%.1f",
            self.doc_count,
            len(self.inverted_index),
            self.avg_doc_len,
        )
        return self

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search corpus using BM25 ranking.
        Returns list of (chunk_id, bm25_score) sorted descending by score.
        """
        if self.doc_count == 0 or not query.strip():
            return []

        query_tokens = tokenize_technical_text(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = {}

        for q in query_tokens:
            if q not in self.inverted_index:
                continue
            idf = self.idf_cache.get(q, 0.0)
            postings = self.inverted_index[q]

            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score = idf * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        if not scores:
            return []

        # Sort doc indices descending by score
        ranked_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.corpus_ids[i], round(scores[i], 4)) for i in ranked_indices]


class BM25IndexManager:
    """
    Singleton manager for the in-memory BM25 index.
    Caches the index and refreshes when documents change.
    """

    _instance: "BM25IndexManager | None" = None

    def __init__(self):
        self._index: BM25Index | None = None
        self._is_dirty: bool = True
        self._total_chunks: int = 0

    @classmethod
    def get_instance(cls) -> "BM25IndexManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def invalidate(self):
        """Mark index as dirty to trigger rebuild on next search."""
        self._is_dirty = True
        logger.info("BM25 index marked dirty (rebuild pending on next query)")

    def get_index(self, collection) -> BM25Index:
        """Return fresh or cached BM25Index from the ChromaDB collection."""
        if self._index is not None and not self._is_dirty:
            return self._index

        logger.info("Building / refreshing BM25 in-memory index from ChromaDB...")
        try:
            # Fetch all documents and IDs from ChromaDB
            results = collection.get(include=["documents"])
            chunk_ids = results.get("ids", [])
            documents = results.get("documents", [])

            index = BM25Index()
            index.fit(chunk_ids, documents)

            self._index = index
            self._is_dirty = False
            self._total_chunks = len(chunk_ids)
            return self._index
        except Exception as exc:
            logger.error("Failed to build BM25 index: %s", exc)
            if self._index is not None:
                return self._index
            return BM25Index()
