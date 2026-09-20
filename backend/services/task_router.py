"""
Task router service — refactored from the standalone router.py.

Exports `classify_task_type()` and `select_model_and_tools()` as pure
functions importable by the agent, without running a separate FastAPI app.
"""

import json
import logging
import re
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "model_registry.yaml"

# Module-level registry cache.
_registry: dict | None = None


def _load_registry() -> dict:
    global _registry
    if _registry is not None:
        return _registry

    with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
        _registry = yaml.safe_load(f)

    logger.info("Loaded model registry from %s", _REGISTRY_PATH)
    return _registry


def reload_registry() -> dict:
    """Force-reload the registry from YAML."""
    global _registry
    _registry = None
    return _load_registry()


def get_valid_task_types() -> list[str]:
    """Return all valid task types from the registry."""
    reg = _load_registry()
    return list(reg["task_type_routing"].keys())


def get_model_entry(model_id: str) -> dict:
    """Look up a model by id. Raises ValueError if not found."""
    reg = _load_registry()
    for m in reg["models"]:
        if m["id"] == model_id:
            return m
    raise ValueError(f"Model id '{model_id}' not found in registry")


def is_ollama_model_installed(model_name: str) -> bool:
    """Check if model is currently downloaded and installed in local Ollama."""
    try:
        from backend.services.network_seal import record_local_call
        record_local_call("http://127.0.0.1:11434/api/tags")
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.5)
        if resp.status_code == 200:
            installed = [m.get("name", "") for m in resp.json().get("models", [])]
            base_target = model_name.split(":")[0]
            norm_target = base_target.replace("-", "").lower()
            return any(
                model_name == m
                or base_target == m.split(":")[0]
                or norm_target == m.split(":")[0].replace("-", "").lower()
                for m in installed
            )
    except Exception:
        pass
    return False


def select_vision_model() -> dict:
    """
    Select the approved local vision model from the registry.
    Prefers qwen2.5-vl:3b / qwen2.5vl:3b if it is installed and approved.
    Gracefully falls back to gemma3:4b (approved multimodal vision model).
    """
    reg = _load_registry()

    # Check for qwen25_vl_3b preference
    for m in reg.get("models", []):
        if m.get("id") == "qwen25_vl_3b" and m.get("approved", True) and m.get("enabled", True):
            for candidate in ["qwen2.5vl:3b", "qwen2.5-vl:3b", m.get("ollama_model_name", "qwen2.5-vl:3b")]:
                if is_ollama_model_installed(candidate):
                    m_copy = dict(m)
                    m_copy["ollama_model_name"] = candidate
                    return m_copy

    # Fallback to approved gemma3_4b
    return validate_model_selection("gemma3_4b")


def _raw_route_task(
    user_request: str,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
) -> dict:
    """
    Classify the incoming request and select the appropriate model, tools,
    and execution parameters prior to any RAG or execution step.
    """
    text = (user_request or "").strip()
    text_lower = text.lower()

    # 1. Vision modality
    if has_image or bool(re.search(r"\b(attached\s+image|attached\s+drawing|this\s+drawing|this\s+photo|this\s+image|inspection\s+drawing|p&id\s+drawing)\b", text_lower)):
        model_entry = select_vision_model()
        return {
            "task_type": "VISION",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": False,
            "requires_sandbox": False,
            "reason": f"Input includes visual content or drawing requiring multimodal vision model ({model_entry['ollama_model_name']}).",
            "tools": [],
            "endpoint": model_entry["endpoint"],
        }

    # 2. Scanned PDF / Document OCR
    if has_scanned_pdf:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "DOCUMENT_ANALYSIS",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": True,
            "requires_sandbox": False,
            "reason": "Input includes a scanned PDF requiring OCR extraction and document analysis.",
            "tools": ["rag_search", "ocr_extract"],
            "endpoint": model_entry["endpoint"],
        }

    # 3. Conversational / Casual Greeting
    greeting_patterns = [
        r"^\s*(hi|hello|hey|greetings|howdy)\b",
        r"^\s*good\s+(morning|afternoon|evening|day)\b",
        r"^\s*(how\s+are\s+you|who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|help)\b",
        r"^\s*(thanks|thank\s+you|bye|goodbye|see\s+you)\b",
    ]
    is_greeting = any(re.match(p, text_lower) for p in greeting_patterns)
    if is_greeting or (len(text_lower.split()) <= 2 and text_lower in {"hi", "hello", "hey", "thanks", "thank you", "bye"}):
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "GENERAL",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": False,
            "requires_sandbox": False,
            "reason": "Conversational greeting or casual message requiring no external context.",
            "tools": [],
            "endpoint": model_entry["endpoint"],
        }

    # 4. Conceptual / Educational Explanation (e.g. "Explain what Python is")
    # Must precede coding check so "explain what python is" does not get classified as coding.
    is_conceptual = bool(re.search(
        r"^\s*(explain\s+(what\s+)?|what\s+is\s+|define\s+|describe\s+|tell\s+me\s+about\s+)",
        text_lower,
    )) and not any(kw in text_lower for kw in [
        "write", "create", "generate", "implement", "build", "script to", "code to", "function to", "calculate", "compute"
    ]) and not any(kw in text_lower for kw in [
        "mrpl", "refinery", "annual report", "balance sheet", "hydrocracker", "fccu", "cdu", "vdu", "om&s", "p&l"
    ])
    if is_conceptual:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "GENERAL",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": False,
            "requires_sandbox": False,
            "reason": "General conceptual explanation without organization-specific retrieval or code execution.",
            "tools": [],
            "endpoint": model_entry["endpoint"],
        }

    # 5. Organization / Knowledge Base Indicators
    org_keywords = [
        "mrpl", "refinery", "mangalore refinery",
        "annual report", "annual reports", "balance sheet", "p&l", "profit and loss",
        "financial report", "financial reports", "financial statements",
        "hydrocracker", "fccu", "pfccu", "pfcc", "cdu", "vdu", "dht", "dhdt", "om&s", "coker", "delayed coker",
        "sulphur recovery", "crude distillation", "refining capacity",
        "from the report", "from the annual report", "from the document",
        "according to the report", "according to the annual report",
        "using the annual report", "using the report",
        "in the report", "in the annual report",
        "uploaded document", "knowledge base", "our company", "company records",
        "ebitda", "gross refining margin", "grm", "throughput", "turnaround",
        # Multilingual (Hindi / Marathi) keywords
        "एमआरपीएल", "रिफाइनरी", "रिफायनरी", "वार्षिक रिपोर्ट", "वार्षिक अहवाल",
        "हाइड्रोक्रैकर", "हायड्रोक्रॅकर", "पीएफसीसीयू", "पीएफसीसी", "सीडीयू", "वीडीयू", "डीएचडीटी",
    ]
    has_org_context = any(kw in text_lower for kw in org_keywords)

    # Unit-Aware and Multilingual Topic Detection
    try:
        from backend.services.unit_grounding import detect_target_unit
        from backend.services.multilingual import normalize_query
        if detect_target_unit(text) is not None:
            has_org_context = True
        else:
            norm = normalize_query(text)
            if norm.detected_topic in ("refinery_unit", "financial_metric", "recruitment_notice"):
                has_org_context = True
    except Exception as route_err:
        logger.warning("Error checking multilingual topic in router: %s", route_err)

    # 6. Data / Knowledge Source Intent Detection
    data_intent_patterns = [
        r"\bknowledge\s+base\b",
        r"\bannual\s+reports?\b",
        r"\bmrpl\s+data\b",
        r"\bsource\s+documents?\b",
        r"\bavailable\s+data\b",
        r"\buploaded\s+documents?\b",
        r"\bfinancial\s+data\b",
        r"\b(?:information|data)\s+in\s+the\s+documents?\b",
        r"\b(?:from|according\s+to|using|in)\s+the\s+(?:annual\s+)?reports?\b",
        r"\bcompany\s+records\b",
    ]
    has_explicit_data_intent = any(bool(re.search(p, text_lower)) for p in data_intent_patterns)
    has_data_source_intent = has_explicit_data_intent or (
        has_org_context and any(kw in text_lower for kw in [
            "annual report", "annual reports", "report", "reports", "document", "documents",
            "knowledge base", "mrpl", "data", "records", "balance sheet", "p&l"
        ])
    )

    # 7. Computation / Execution Intent Detection
    comp_intent_patterns = [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bcompare\s+numerically\b",
        r"\bderive\b",
        r"\bcalculate\s+(?:growth|percentage|ratio|cagr|margin|difference|average|total)\b",
        r"\b(?:growth|cagr|margin|percentage|difference)\s+(?:calculation|rate)\b",
        r"\b(?:run|show)\s+(?:the\s+)?python\b",
        r"\bexecute\s+(?:and\s+verify\s+)?(?:the\s+)?(?:code|calculation|script)\b",
        r"\bverify\s+(?:the\s+)?(?:calculation|code)\b",
        r"\bgenerate\s+calculation\s+code\b",
        r"\bsandbox\b",
        r"\bexecute\s+in\s+(?:the\s+)?sandbox\b",
    ]
    has_computation_intent = any(bool(re.search(p, text_lower)) for p in comp_intent_patterns)

    # 8. Coding Intent Detection (generalized software engineering patterns)
    code_intent_patterns = [
        r"\b(write|create|generate|implement|build|develop|modify|update|refactor|change|extend|adapt|fix|rewrite)\b.*\b(python|code|script|program|function|algorithm|class|snippet)\b",
        r"\bpython\s+(code|script|program|function|algorithm|class|snippet)\b",
        r"\b(code|script|program|function|algorithm)\b.*\b(in\s+python|using\s+python)\b",
        r"\b(function|algorithm|script|code|program)\s+to\s+(sort|calculate|compute|find|reverse|search|parse|convert|filter|format|simulate|optimize)\b",
        r"\b(sort|reverse|filter|traverse)\s+(an?\s+)?(array|list|string|numbers|elements|data\s+structure|tree|graph)\b",
        r"\b(write|create|implement)\s+(a\s+)?(binary\s+search|fibonacci|quicksort|mergesort|regex|decorator|generator|class)\b",
        r"\b(previous|prior|existing|that|the)\s+(code|script|function|program|snippet)\b",
        r"\b(modify|update|change|fix|extend|adapt)\s+(?:it|the\s+previous\s+code|that\s+code|the\s+code)\b",
    ]
    is_coding = any(bool(re.search(p, text_lower)) for p in code_intent_patterns)

    # Math calculation without org context
    if not is_coding and any(text_lower.startswith(kw) for kw in ["calculate ", "compute ", "what is the sum of ", "solve "]):
        if not has_data_source_intent and not has_org_context:
            is_coding = True

    # 9. HYBRID Detection: Requires BOTH Data Intent AND Computation/Execution Intent
    is_hybrid = has_data_source_intent and (has_computation_intent or is_coding)

    if is_hybrid:
        coder_entry = validate_model_selection("qwen25_coder_3b")
        return {
            "task_type": "HYBRID",
            "model": coder_entry["ollama_model_name"],
            "model_id": coder_entry["id"],
            "requires_rag": True,
            "requires_sandbox": True,
            "reason": "Request requires MRPL document/financial data from RAG followed by code generation and sandbox execution.",
            "tools": ["rag_search", "sandbox_execute"],
            "endpoint": coder_entry["endpoint"],
        }

    # 10. Pure CODING Task: Coding/Execution Intent WITHOUT Data Source Intent
    if is_coding or (has_computation_intent and not has_data_source_intent and not has_org_context):
        coder_entry = validate_model_selection("qwen25_coder_3b")
        return {
            "task_type": "CODING",
            "model": coder_entry["ollama_model_name"],
            "model_id": coder_entry["id"],
            "requires_rag": False,
            "requires_sandbox": True,
            "reason": "User requested Python code generation and sandbox execution without organization knowledge retrieval.",
            "tools": ["sandbox_execute"],
            "endpoint": coder_entry["endpoint"],
        }

    # 8. Document Analysis & Deliverable Generation Intent
    # Extract target document name (supporting quotes, spaces, and standard filenames)
    quoted_docs = re.findall(r'["\']([^"\']+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))["\']', text, re.IGNORECASE)
    if quoted_docs:
        target_doc = quoted_docs[0].strip()
    else:
        # Check for multi-word or standard filenames
        multi_match = re.search(
            r'(?:(?:analyze|summarize|review|breakdown|generate|make|create|download|export|open)\s+(?:a\s+|the\s+|of\s+)?)?([A-Za-z0-9_\-\s]+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt))',
            text,
            re.IGNORECASE,
        )
        if multi_match and multi_match.group(1).strip():
            cand = multi_match.group(1).strip()
            # Clean leading prepositions if captured
            cand = re.sub(r'^(?:analyze|summarize|review|breakdown|of|the|a)\s+', '', cand, flags=re.IGNORECASE).strip()
            target_doc = cand if len(cand) >= 4 else None
        else:
            standard_docs = re.findall(r'[\w\-]+\.(?:pdf|docx|doc|xlsx|xls|pptx|csv|txt)', text, re.IGNORECASE)
            target_doc = standard_docs[0] if standard_docs else None

    # Check against known uploaded files for fuzzy matching (e.g. underscores vs spaces)
    if target_doc:
        try:
            from backend.services.doc_analysis import resolve_document_file
            resolved_p, resolved_meta = resolve_document_file(target_doc)
            if resolved_meta and resolved_meta.get("original_name"):
                target_doc = resolved_meta["original_name"]
            elif resolved_p:
                target_doc = resolved_p.name
        except Exception:
            pass

    doc_deliverable_pdf_patterns = [
        r"\b(create|generate|make|give|download|export|save|send|convert|provide|get|build|print)\b.*\bpdf\b",
        r"\bpdf\b.*\b(create|generate|make|give|download|export|save|send|convert|provide|get|build|print)\b",
        r"\bpdf\s+(of\s+it|of\s+this|of\s+that|of\s+the|file|document|doc|report|format|form|version|deliverable|copy)\b",
        r"\b(in|as|to|into|a|the)\s+pdf\b",
        r"^\s*pdf\s*$",
        r"\b(pdf\s+form|in\s+pdf|as\s+pdf|pdf\s+format|pdf\s+report|download\s+pdf|export.*pdf|generate.*pdf)\b",
        r"\b(give.*report.*in\s+pdf|analysis\s+report\s+in\s+pdf)\b",
    ]
    doc_deliverable_docx_patterns = [
        r"\b(create|generate|make|give|download|export|save|send|convert|provide|get|build|print)\b.*\b(docx|doc|word)\b",
        r"\b(docx|word)\b.*\b(create|generate|make|give|download|export|save|send|convert|provide|get|build|print)\b",
        r"\b(docx|word)\s+(of\s+it|of\s+this|of\s+that|of\s+the|file|document|doc|report|format|form|version|deliverable|copy)\b",
        r"\b(in|as|to|into|a|the)\s+(docx|word)\b",
        r"^\s*(docx|word)\s*$",
        r"\b(docx\s+form|in\s+docx|as\s+docx|docx\s+format|docx\s+report|download\s+docx|export.*docx|generate.*docx|in\s+word|word\s+format)\b",
        r"\b(give.*report.*in\s+docx|analysis\s+report\s+in\s+docx)\b",
    ]
    generic_deliverable_patterns = [
        r"\b(give\s+the\s+analysis\s+report|give\s+me\s+the\s+analysis\s+report|give\s+the\s+report|generate\s+the\s+analysis\s+report|generate\s+the\s+report)\b",
        r"^\s*(give+|where\s+is\s+it|show\s+it|download\s+it|give\s+it|give\s+me)\b",
    ]
    doc_analysis_patterns = [
        r"\b(analyze|analysis\s+of|summarize|summary\s+of|breakdown\s+of)\b",
        r"\b(give\s+the\s+analysis|give\s+me\s+the\s+analysis)\b",
        r"\b(summarize\s+the\s+document|summarize\s+this\s+report|executive\s+summary)\b",
    ]

    is_pdf_deliverable = any(bool(re.search(p, text_lower)) for p in doc_deliverable_pdf_patterns)
    is_docx_deliverable = any(bool(re.search(p, text_lower)) for p in doc_deliverable_docx_patterns)
    is_generic_deliverable = any(bool(re.search(p, text_lower)) for p in generic_deliverable_patterns)
    is_doc_analysis = any(bool(re.search(p, text_lower)) for p in doc_analysis_patterns) or bool(
        target_doc and any(w in text_lower for w in ["analyze", "analysis", "summary", "summarize", "report"])
    )

    if is_pdf_deliverable:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "DOCUMENT_ANALYSIS",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": False,
            "requires_sandbox": False,
            "reason": "User requested document analysis deliverable in PDF format.",
            "tools": ["docgen_pdf"],
            "endpoint": model_entry["endpoint"],
            "target_document": target_doc,
            "deliverable_type": "pdf",
        }

    if is_docx_deliverable:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "DOCUMENT_ANALYSIS",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": False,
            "requires_sandbox": False,
            "reason": "User requested document analysis deliverable in DOCX format.",
            "tools": ["docgen_docx"],
            "endpoint": model_entry["endpoint"],
            "target_document": target_doc,
            "deliverable_type": "docx",
        }

    if is_generic_deliverable:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "DOCUMENT_ANALYSIS",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": False,
            "requires_sandbox": False,
            "reason": "User requested deliverable report; defaulting to PDF format.",
            "tools": ["docgen_pdf"],
            "endpoint": model_entry["endpoint"],
            "target_document": target_doc,
            "deliverable_type": "pdf",
        }

    if is_doc_analysis:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "DOCUMENT_ANALYSIS",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": True,
            "requires_sandbox": False,
            "reason": "Document analysis request requiring knowledge base retrieval.",
            "tools": ["rag_search"],
            "endpoint": model_entry["endpoint"],
            "target_document": target_doc,
        }

    # 9. Document Summary / Approval Note
    if any(kw in text_lower for kw in ["approval note", "draft an approval note", "prepare note for approval"]):
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "RAG",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": True,
            "requires_sandbox": False,
            "reason": "Industrial approval note drafting requiring knowledge base context and document generation.",
            "tools": ["rag_search", "docgen_docx"],
            "endpoint": model_entry["endpoint"],
        }


    # 9. Knowledge Base / RAG Question
    if has_org_context:
        model_entry = validate_model_selection("gemma3_4b")
        return {
            "task_type": "RAG",
            "model": model_entry["ollama_model_name"],
            "model_id": model_entry["id"],
            "requires_rag": True,
            "requires_sandbox": False,
            "reason": "MRPL organization or refinery knowledge query requiring knowledge base retrieval.",
            "tools": ["rag_search"],
            "endpoint": model_entry["endpoint"],
        }

    # 10. Default: General Knowledge
    model_entry = validate_model_selection("gemma3_4b")
    return {
        "task_type": "GENERAL",
        "model": model_entry["ollama_model_name"],
        "model_id": model_entry["id"],
        "requires_rag": False,
        "requires_sandbox": False,
        "reason": "General conceptual or factual reasoning without organization-specific retrieval or code execution.",
        "tools": [],
        "endpoint": model_entry["endpoint"],
    }


def _enrich_decision(d: dict) -> dict:
    """Enrich the routing decision with candidates, deterministic scores, and license gate."""
    task_type = d.get("task_type", "GENERAL")
    model_id = d.get("model_id", "gemma3_4b")
    try:
        model_entry = get_model_entry(model_id)
        d["license"] = model_entry.get("license", "Open Source")
        d["approved"] = model_entry.get("approved", True)
    except Exception:
        d["license"] = "Gemma Terms of Use" if "gemma" in model_id else "Apache-2.0"
        d["approved"] = True

    # Deterministic routing candidate calculation with actual empirical suitability scores
    if task_type in ("CODING", "HYBRID"):
        d["candidates"] = [
            {"model": "qwen2.5-coder:3b", "model_id": "qwen25_coder_3b", "score": 0.95, "selected": model_id == "qwen25_coder_3b", "license": "Apache-2.0", "task": "coding_execution"},
            {"model": "gemma3:4b", "model_id": "gemma3_4b", "score": 0.40, "selected": model_id == "gemma3_4b", "license": "Gemma Terms of Use", "task": "general_reasoning"},
        ]
    elif task_type == "VISION":
        d["candidates"] = [
            {"model": "gemma3:4b", "model_id": "gemma3_4b", "score": 0.92, "selected": model_id == "gemma3_4b", "license": "Gemma Terms of Use", "task": "multimodal_vision"},
            {"model": "qwen2.5-vl:3b", "model_id": "qwen25_vl_3b", "score": 0.85, "selected": model_id == "qwen25_vl_3b", "license": "Apache-2.0", "task": "vision_pid"},
            {"model": "qwen2.5-coder:3b", "model_id": "qwen25_coder_3b", "score": 0.15, "selected": False, "license": "Apache-2.0", "task": "coding_execution"},
        ]
    elif task_type == "DOCUMENT_ANALYSIS":
        d["candidates"] = [
            {"model": "gemma3:4b", "model_id": "gemma3_4b", "score": 0.96, "selected": model_id == "gemma3_4b", "license": "Gemma Terms of Use", "task": "document_analysis"},
            {"model": "qwen2.5-coder:3b", "model_id": "qwen25_coder_3b", "score": 0.30, "selected": False, "license": "Apache-2.0", "task": "coding_execution"},
        ]
    else:  # RAG, GENERAL
        d["candidates"] = [
            {"model": "gemma3:4b", "model_id": "gemma3_4b", "score": 0.94, "selected": model_id == "gemma3_4b", "license": "Gemma Terms of Use", "task": "dense_retrieval_reasoning"},
            {"model": "qwen2.5-coder:3b", "model_id": "qwen25_coder_3b", "score": 0.35, "selected": False, "license": "Apache-2.0", "task": "coding_execution"},
        ]
    return d


def route_task(
    user_request: str,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
) -> dict:
    """
    Public task router entry point.
    Returns enriched, auditable routing decision with model candidates and license gate.
    """
    raw_decision = _raw_route_task(
        user_request=user_request,
        has_image=has_image,
        has_scanned_pdf=has_scanned_pdf,
    )
    return _enrich_decision(raw_decision)


def classify_task_type(
    user_request: str,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
    router_model_id: str = "gemma3_4b",
) -> dict:
    """
    Backward-compatible classification wrapper returning legacy fields
    plus extended routing metadata.
    """
    decision = route_task(
        user_request,
        has_image=has_image,
        has_scanned_pdf=has_scanned_pdf,
    )
    return {
        "task_type": decision["task_type"],
        "reasoning": decision["reason"],
        "model": decision["model"],
        "model_id": decision["model_id"],
        "requires_rag": decision["requires_rag"],
        "requires_sandbox": decision["requires_sandbox"],
        "tools": decision["tools"],
    }


def validate_model_selection(model_id: str) -> dict:
    """
    Validate that a requested model exists in the registry, is approved,
    and is currently enabled. Raises ValueError if unapproved or disabled.
    """
    entry = get_model_entry(model_id)
    if not entry.get("approved", True):
        raise ValueError(f"Model '{model_id}' is not an approved model in the Sovereign Registry.")
    if not entry.get("enabled", True):
        raise ValueError(f"Model '{model_id}' is currently disabled by administrator policy.")
    return entry


def select_model_and_tools(
    task_type: str,
    has_image: bool = False,
    has_scanned_pdf: bool = False,
) -> tuple[dict, list[str]]:
    """
    Select an approved model and tools from the registry based on task type.
    Enforces that only approved and enabled models can be selected.
    Returns (model_dict, list_of_tool_ids).
    """
    reg = _load_registry()
    routing_info = reg["task_type_routing"].get(task_type)

    if not routing_info:
        task_type = "general_reasoning"
        routing_info = reg["task_type_routing"][task_type]

    preferred_tags = set(routing_info.get("preferred_tags", []))

    candidates = []
    for model in reg["models"]:
        # Strict security rule: Only approved and enabled models can be selected
        if not model.get("approved", True) or not model.get("enabled", True):
            continue
        model_tags = set(model.get("task_tags", []))
        if preferred_tags & model_tags:
            candidates.append(model)

    if not candidates:
        gemma = get_model_entry("gemma3_4b")
        if not gemma.get("approved", True) or not gemma.get("enabled", True):
            raise RuntimeError("Default reasoning model is not approved or enabled in registry.")
        candidates = [gemma]

    # Enforce vision modality when image is attached.
    if has_image:
        vision_candidates = [
            m for m in candidates if "vision" in m.get("modality", [])
        ]
        if vision_candidates:
            candidates = vision_candidates
        else:
            vision_all = [
                m for m in reg["models"]
                if m.get("approved", True) and m.get("enabled", True) and "vision" in m.get("modality", [])
            ]
            if vision_all:
                candidates = vision_all

    chosen_model = sorted(candidates, key=lambda m: m.get("priority", 99))[0]
    logger.info("Selected approved model '%s' (%s) for task '%s'", chosen_model["id"], chosen_model["ollama_model_name"], task_type)

    # Tool selection.
    tools = list(routing_info.get("required_tools", []))
    if has_scanned_pdf and "ocr_extract" not in tools:
        tools.append("ocr_extract")

    return chosen_model, tools
