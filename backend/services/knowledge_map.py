"""
Knowledge Map & Knowledge Base Service.

Authoritative aggregation of the persistent ChromaDB collection ('mrpl_knowledge_base')
used by the RAG engine, incorporating RBAC clearance and document classification.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.rag_config import EMBEDDING_MODEL_NAME
from backend.services.rag_engine import (
    CLASSIFICATION_LEVELS,
    clean_document_title,
    get_collection,
    get_max_clearance_for_roles,
)

logger = logging.getLogger(__name__)

# Category display names & descriptions for the MRPL corpus
CATEGORY_METADATA = {
    "01_Refinery_Manufacturing": {
        "name": "Refining & Manufacturing",
        "description": "Refinery process units (CDU, HCU, PFCCU), polypropylene manufacturing, operational cutting edge technology.",
        "icon": "refinery",
        "default_classification": "INTERNAL",
    },
    "02_Environment_Compliance": {
        "name": "Environment & Compliance",
        "description": "Environmental clearances (EC), CRZ compliance, KSPCB/MoEF filings, desalination and effluent management.",
        "icon": "environment",
        "default_classification": "INTERNAL",
    },
    "03_Safety_HSE": {
        "name": "Health, Safety & Environment (HSE)",
        "description": "Industrial safety guidelines, CVC vigilance regulations, PIDPI awareness, and workplace health standards.",
        "icon": "safety",
        "default_classification": "INTERNAL",
    },
    "04_Company_General": {
        "name": "Corporate & General",
        "description": "Corporate overview, public notices, organizational structure, and vision statements.",
        "icon": "corporate",
        "default_classification": "PUBLIC",
    },
    "05_Policies_Certifications": {
        "name": "Policies & Certifications",
        "description": "AS 9100D, ISO standards, information security policy, and internal quality audit reports.",
        "icon": "policy",
        "default_classification": "INTERNAL",
    },
    "06_Website_Content": {
        "name": "Public Portal & Multilingual",
        "description": "Bilingual website content, capacity statements, Hindi/English corporate literature.",
        "icon": "portal",
        "default_classification": "PUBLIC",
    },
    "07_Finance": {
        "name": "Finance & Annual Reports",
        "description": "Audited annual reports (FY 2014 to FY 2026), financial statements, SEBI filings, and subsidiary accounts.",
        "icon": "finance",
        "default_classification": "INTERNAL",
    },
    "08_Other": {
        "name": "Technical & Regulatory Filings",
        "description": "Station consent orders (APMC, Sarpadi), technical permits, and regulatory disclosures.",
        "icon": "technical",
        "default_classification": "INTERNAL",
    },
    "09_Confidential": {
        "name": "Confidential Engineering & Strategy",
        "description": "Piping & Instrumentation Diagrams (P&IDs), vendor negotiations, internal correspondence, unreleased designs.",
        "icon": "confidential",
        "default_classification": "CONFIDENTIAL",
    },
    "general": {
        "name": "Operational Manuals & Uploads",
        "description": "User-uploaded technical manuals, vigilance portal documentation, and verified system reports.",
        "icon": "manuals",
        "default_classification": "INTERNAL",
    },
    "root": {
        "name": "Engineering Reports & Coverage",
        "description": "System extraction manifests, OCR coverage summaries, and reorg plans.",
        "icon": "manifest",
        "default_classification": "INTERNAL",
    },
}

# In-memory cache for raw aggregated documents from ChromaDB
_raw_cache: dict[str, Any] | None = None
_raw_cache_time: float = 0.0
_CACHE_TTL_SECONDS = 300.0  # 5 minutes, invalidated upon upload/reindex


def invalidate_knowledge_map_cache() -> None:
    """Invalidate in-memory cache when new files are uploaded or reindexed."""
    global _raw_cache, _raw_cache_time
    _raw_cache = None
    _raw_cache_time = 0.0
    logger.info("Knowledge map cache invalidated.")


def _get_raw_corpus_data() -> dict[str, dict]:
    """
    Fetch all unique document records from the persistent ChromaDB collection.
    Cached for fast sub-millisecond responses.
    """
    global _raw_cache, _raw_cache_time
    now = time.time()
    if _raw_cache is not None and (now - _raw_cache_time) < _CACHE_TTL_SECONDS:
        return _raw_cache

    collection = get_collection()
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas", []) or []

    # Aggregate by unique document source
    docs_by_source: dict[str, dict] = defaultdict(lambda: {
        "chunks": 0,
        "pages": set(),
        "categories": set(),
        "document_types": set(),
        "years": set(),
        "paths": set(),
        "classifications": set(),
        "departments": set(),
        "owners": set(),
        "versions": set(),
        "file_ids": set(),
    })

    for m in metadatas:
        if not m:
            continue
        source = m.get("source") or Path(m.get("path", "unknown")).name
        docs_by_source[source]["chunks"] += 1

        if m.get("page") is not None:
            docs_by_source[source]["pages"].add(m.get("page"))
        if m.get("category"):
            docs_by_source[source]["categories"].add(m.get("category"))
        if m.get("document_type"):
            docs_by_source[source]["document_types"].add(m.get("document_type"))
        if m.get("year") and m.get("year") != "Unknown":
            docs_by_source[source]["years"].add(m.get("year"))
        if m.get("path"):
            docs_by_source[source]["paths"].add(m.get("path"))
        if m.get("classification"):
            docs_by_source[source]["classifications"].add(str(m.get("classification")).upper())
        if m.get("department"):
            docs_by_source[source]["departments"].add(m.get("department"))
        if m.get("owner"):
            docs_by_source[source]["owners"].add(m.get("owner"))
        if m.get("version"):
            docs_by_source[source]["versions"].add(m.get("version"))
        if m.get("file_id"):
            docs_by_source[source]["file_ids"].add(m.get("file_id"))

    # Lookup original names from uploaded files in SQLite if available
    try:
        from backend.database.repositories import files as files_repo
        uploaded = files_repo.list_uploaded_files()
        uploaded_map = {Path(f["stored_path"]).name: f["original_name"] for f in uploaded if f.get("stored_path")}
        uploaded_id_map = {f["id"]: f["original_name"] for f in uploaded if f.get("id")}
    except Exception:
        uploaded_map = {}
        uploaded_id_map = {}

    # Convert sets to serializable lists/values
    processed: dict[str, dict] = {}
    for source, info in docs_by_source.items():
        # Determine primary category
        cat = next(iter(info["categories"])) if info["categories"] else "general"
        if cat == "manufacturing":
            cat = "01_Refinery_Manufacturing"

        doc_type = next(iter(info["document_types"])) if info["document_types"] else "pdf"

        # Determine display filename
        display_name = source
        file_id = next(iter(info["file_ids"])) if info["file_ids"] else None
        if source in uploaded_map:
            display_name = uploaded_map[source]
        elif file_id and file_id in uploaded_id_map:
            display_name = uploaded_id_map[file_id]

        # Determine classification
        if info["classifications"]:
            classification = next(iter(info["classifications"]))
        elif "09_Confidential" in cat:
            classification = "CONFIDENTIAL"
        elif cat in ("04_Company_General", "06_Website_Content"):
            classification = "PUBLIC"
        else:
            classification = CATEGORY_METADATA.get(cat, {}).get("default_classification", "INTERNAL")

        # Check detected refinery units in filename and title
        source_lower = f"{display_name} {source}".lower()
        units = []
        if "pfccu" in source_lower or "polypropylene" in source_lower:
            units.append("PFCCU")
        if "hydrocracker" in source_lower or "hcu" in source_lower:
            units.append("HCU")
        if "cdu" in source_lower or "crude" in source_lower:
            units.append("CDU")
        if "desal" in source_lower:
            units.append("DESAL")
        if "p&id" in source_lower:
            units.append("P&ID")

        processed[source] = {
            "source": display_name,
            "raw_source": source,
            "title": clean_document_title(display_name),
            "category": cat,
            "category_name": CATEGORY_METADATA.get(cat, {}).get("name", cat.replace("_", " ").title()),
            "document_type": doc_type,
            "chunks_count": info["chunks"],
            "page_count": len(info["pages"]) if info["pages"] else None,
            "classification": classification,
            "department": next(iter(info["departments"])) if info["departments"] else "GENERAL",
            "year": next(iter(info["years"])) if info["years"] else None,
            "path": next(iter(info["paths"])) if info["paths"] else None,
            "file_id": file_id,
            "units_detected": units,
            "indexed_status": "indexed",
        }

    _raw_cache = processed
    _raw_cache_time = now
    return processed


def get_knowledge_map_data(user_roles: list[str] | None = None, user_clearance: str | None = None) -> dict[str, Any]:
    """
    Generate the structured Knowledge Map hierarchy respecting RBAC and user clearance.
    Authorized users see all authorized documents and branches.
    Unauthorized users cannot view confidential resources.
    """
    corpus = _get_raw_corpus_data()

    # Determine user clearance level
    clearance = user_clearance or get_max_clearance_for_roles(user_roles)
    max_level = CLASSIFICATION_LEVELS.get(str(clearance).upper(), 1)

    categories_map: dict[str, dict] = {}
    authorized_docs_count = 0
    authorized_chunks_count = 0

    for source, doc in corpus.items():
        doc_class = str(doc.get("classification", "INTERNAL")).upper()
        doc_level = CLASSIFICATION_LEVELS.get(doc_class, 2)
        is_accessible = doc_level <= max_level

        # Strict security rule: If user does not have clearance for CONFIDENTIAL/HIGHLY_CONFIDENTIAL,
        # exclude confidential asset metadata to prevent unauthorized disclosure.
        if not is_accessible and doc_level >= 3:
            continue

        cat_key = doc["category"]
        if cat_key not in categories_map:
            cat_meta = CATEGORY_METADATA.get(cat_key, {})
            categories_map[cat_key] = {
                "id": f"cat_{cat_key}",
                "category_code": cat_key,
                "name": cat_meta.get("name", cat_key.replace("_", " ").title()),
                "description": cat_meta.get("description", ""),
                "icon": cat_meta.get("icon", "folder"),
                "classification": cat_meta.get("default_classification", "INTERNAL"),
                "document_count": 0,
                "total_chunks": 0,
                "documents": [],
            }

        cat_entry = categories_map[cat_key]
        cat_entry["document_count"] += 1
        cat_entry["total_chunks"] += doc["chunks_count"]
        authorized_docs_count += 1
        authorized_chunks_count += doc["chunks_count"]

        # Add document to category list
        cat_entry["documents"].append({
            "id": f"doc_{abs(hash(source)) % 1000000}",
            "filename": doc["source"],
            "title": doc["title"],
            "category": doc["category"],
            "category_name": doc["category_name"],
            "document_type": doc["document_type"],
            "page_count": doc["page_count"],
            "chunks_count": doc["chunks_count"],
            "classification": doc["classification"],
            "department": doc["department"],
            "indexed_status": "indexed",
            "accessible": is_accessible,
            "year": doc["year"],
            "units_detected": doc["units_detected"],
            "file_id": doc["file_id"],
        })

    # Sort categories in logical order
    category_order = [
        "01_Refinery_Manufacturing",
        "02_Environment_Compliance",
        "03_Safety_HSE",
        "09_Confidential",
        "07_Finance",
        "05_Policies_Certifications",
        "04_Company_General",
        "06_Website_Content",
        "08_Other",
        "general",
        "root",
    ]

    sorted_categories = []
    for code in category_order:
        if code in categories_map:
            cat = categories_map.pop(code)
            # Sort documents by chunks descending
            cat["documents"].sort(key=lambda x: -x["chunks_count"])
            sorted_categories.append(cat)

    # Append any remaining categories
    for code, cat in sorted(categories_map.items()):
        cat["documents"].sort(key=lambda x: -x["chunks_count"])
        sorted_categories.append(cat)

    return {
        "root": {
            "id": "root_kb",
            "name": "MRPL Sovereign Knowledge Base",
            "total_documents": authorized_docs_count,
            "total_chunks": authorized_chunks_count,
            "total_categories": len(sorted_categories),
            "indexed_status": "Fully Indexed (ChromaDB)",
            "embedding_engine": f"{EMBEDDING_MODEL_NAME} (Local CPU)",
            "user_clearance": str(clearance).upper(),
            "max_clearance_level": max_level,
        },
        "categories": sorted_categories,
    }
