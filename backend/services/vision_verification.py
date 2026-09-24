"""
P&ID and Engineering Drawing Vision Verification Service.

Orchestrates local Ollama multimodal vision analysis for P&ID schematics and refinery diagrams.
Deterministically validates equipment tags against visual OCR/image evidence first, ensuring
no equipment tags are hallucinated from domain knowledge or registries.
Separates findings into 5 structured sections:
1. Visually observed elements
2. Process interpretation
3. Verifiable equipment tags
4. Registry verification
5. Uncertainty / review required
"""

import base64
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any

import requests

from backend.database.repositories import approvals as approvals_repo
from backend.services.audit import audit_log
from backend.services.network_seal import record_local_call

logger = logging.getLogger(__name__)

# ── Vision Analysis Status Constants ─────────────────────────────────────
VISION_STATUS_STARTED = "VISION_ANALYSIS_STARTED"
VISION_STATUS_MODEL_SELECTED = "VISION_MODEL_SELECTED"
VISION_STATUS_COMPLETED = "VISION_ANALYSIS_COMPLETED"
VISION_STATUS_INFERENCE_FAILED = "VISION_INFERENCE_FAILED"
VISION_STATUS_RETRY_STARTED = "VISION_RETRY_STARTED"
VISION_STATUS_OCR_FALLBACK = "VISION_OCR_FALLBACK"
VISION_STATUS_FAILED = "VISION_ANALYSIS_FAILED"
VISION_STATUS_RELEVANCE_CHECKED = "VISION_RELEVANCE_CHECKED"
VISION_STATUS_REQUIRES_REVIEW = "VISION_REQUIRES_REVIEW"
VISION_STATUS_REPORT_GEN_STARTED = "VISION_REPORT_GENERATION_STARTED"
VISION_STATUS_REPORT_GENERATED = "VISION_REPORT_GENERATED"
PDF_STATUS_GEN_STARTED = "PDF_GENERATION_STARTED"
PDF_STATUS_GENERATED = "PDF_GENERATED"
PDF_STATUS_RETURNED = "PDF_RETURNED"
VISION_STATUS_NOT_APPLICABLE = "VISION_NOT_APPLICABLE"

# Verification result statuses (expanded from original)
VERIFIED_NO_TAGS = "VERIFIED_NO_TAGS"
VERIFIED_TAGS_FOUND = "VERIFIED_TAGS_FOUND"
REQUIRES_REVIEW = "REQUIRES_REVIEW"

# ── Vision-Specific Generation Configuration ─────────────────────────────
# Conservative settings to prevent token repeat limit errors with qwen2.5-vl:3b.
# These do NOT affect non-vision (text/RAG) tasks.
VISION_GENERATION_OPTIONS = {
    "temperature": 0.05,
    "top_p": 0.8,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "num_predict": 512,
}

VISION_RETRY_OPTIONS = {
    "temperature": 0.05,
    "top_p": 0.8,
    "repeat_penalty": 1.2,
    "repeat_last_n": 64,
    "num_predict": 256,
}

# Optical transcription options (used internally for visual evidence extraction)
VISION_OCR_TRANSCRIPTION_OPTIONS = {
    "temperature": 0.0,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "num_predict": 300,
}


@dataclass
class VisionInferenceResult:
    """Structured result from a vision model inference call.

    Explicitly captures success vs. failure so downstream consumers
    can never mistake a failed inference for a successful analysis
    that found no content.
    """
    success: bool
    text: str = ""
    error: str = ""
    error_code: str = ""     # "HTTP_500", "TIMEOUT", "TOKEN_REPEAT_LIMIT", "PREDICTION_ABORTED", etc.
    source: str = ""         # "vision_model", "ocr", "combined", "retry"
    retried: bool = False
    analysis_id: str = ""
    image_hash: str = ""
    ocr_text: str = ""       # OCR-extracted text (if any), separate from vision
    vision_model_text: str = ""  # Raw vision model output (if any)


def is_vision_failure_response(status_code: int, response_text: str) -> bool:
    """Detect whether a vision model response indicates a generation failure.

    Primary detection:
    - HTTP status != 200 (covers 500, 404, etc.)
    - Empty or whitespace-only model output
    - Known Ollama error patterns in response body

    Defensive fallback:
    - Response body containing error-like JSON patterns
    """
    # Primary: HTTP status
    if status_code != 200:
        return True

    text_lower = (response_text or "").lower().strip()

    # Primary: empty output
    if not text_lower:
        return True

    # Primary: known Ollama error patterns
    failure_patterns = [
        "token repeat limit",
        "prediction aborted",
        "model not found",
        "internal server error",
        "out of memory",
        "error loading model",
        "context window exceeded",
        '"error"',  # JSON error envelope from Ollama
    ]
    if any(p in text_lower for p in failure_patterns):
        return True

    return False


def _classify_vision_error(status_code: int, response_text: str) -> str:
    """Classify the type of vision inference error for tracing."""
    text_lower = (response_text or "").lower()
    if "token repeat limit" in text_lower:
        return "TOKEN_REPEAT_LIMIT"
    if "prediction aborted" in text_lower:
        return "PREDICTION_ABORTED"
    if status_code == 500 or "internal server error" in text_lower:
        return "HTTP_500"
    if status_code == 404 or "model not found" in text_lower:
        return "MODEL_NOT_FOUND"
    if "timeout" in text_lower or "timed out" in text_lower:
        return "TIMEOUT"
    if "out of memory" in text_lower:
        return "OUT_OF_MEMORY"
    if not (response_text or "").strip():
        return "EMPTY_OUTPUT"
    return f"HTTP_{status_code}"


def build_short_vision_prompt(user_query: str) -> str:
    """Build a short, structured prompt for vision tasks.

    Designed to prevent token repetition by requesting concise output
    in a numbered list format. Does NOT ask for lengthy analysis.
    """
    return (
        "Describe this diagram or schematic concisely. Provide:\n"
        "1. Overall diagram purpose (1 sentence)\n"
        "2. Main visible components / unit labels\n"
        "3. Flow direction and connections\n"
        "4. Major stages or architecture layers\n"
        "5. Clearly visible alphanumeric tags or IDs (if any)\n"
        "6. Uncertain or unreadable labels\n\n"
        "Do not repeat information. Do not invent labels not visible in the image. "
        "Describe ONLY what can be visually supported.\n"
        f"User question: {user_query}"
    )

# Known Authoritative MRPL Refinery Equipment Register (from Knowledge Base / P&IDs)
KNOWN_EQUIPMENT_REGISTER = {
    # Crude Distillation Unit (CDU / VDU)
    "11-C-101": {"name": "Atmospheric Distillation Column", "unit": "CDU-1", "service": "Crude Fractionation", "status": "ACTIVE"},
    "11-V-101": {"name": "Electrostatic Desalter", "unit": "CDU-1", "service": "Crude Desalting", "status": "ACTIVE"},
    "11-P-101A": {"name": "Crude Charge Pump (Motor Driven)", "unit": "CDU-1", "service": "Crude Transfer", "status": "ACTIVE"},
    "11-P-101B": {"name": "Crude Charge Pump (Turbine Driven)", "unit": "CDU-1", "service": "Crude Transfer", "status": "ACTIVE"},
    "11-E-101A": {"name": "Crude / Kerosene Heat Exchanger", "unit": "CDU-1", "service": "Preheat Train", "status": "ACTIVE"},
    "11-E-101B": {"name": "Crude / Diesel Heat Exchanger", "unit": "CDU-1", "service": "Preheat Train", "status": "ACTIVE"},
    "11-F-101": {"name": "Crude Charge Heater Furnace", "unit": "CDU-1", "service": "Crude Heating", "status": "ACTIVE"},
    "12-C-101": {"name": "Vacuum Distillation Column", "unit": "VDU", "service": "Heavy Ends Separation", "status": "ACTIVE"},

    # Hydrocracker Unit (HCU)
    "21-R-101": {"name": "Hydrocracker First Stage Reactor", "unit": "HCU", "service": "Hydrocracking", "status": "ACTIVE"},
    "21-R-102": {"name": "Hydrocracker Second Stage Reactor", "unit": "HCU", "service": "Hydrocracking", "status": "ACTIVE"},
    "21-K-101A": {"name": "Recycle Gas Compressor A", "unit": "HCU", "service": "Hydrogen Circulation", "status": "ACTIVE"},
    "21-P-101A": {"name": "High Pressure Feed Pump", "unit": "HCU", "service": "Reactor Feed", "status": "ACTIVE"},

    # Petrochemical FCCU (PFCCU)
    "31-R-101": {"name": "PFCCU Riser Reactor", "unit": "PFCCU", "service": "Catalytic Cracking", "status": "ACTIVE"},
    "31-R-102": {"name": "PFCCU Catalyst Regenerator", "unit": "PFCCU", "service": "Catalyst Regeneration", "status": "ACTIVE"},
    "31-K-101": {"name": "Main Air Blower", "unit": "PFCCU", "service": "Combustion Air", "status": "ACTIVE"},
    "31-C-101": {"name": "Main Fractionator Column", "unit": "PFCCU", "service": "Cracked Products Fractionation", "status": "ACTIVE"},
    "31-P-101A": {"name": "Slurry Circulation Pump", "unit": "PFCCU", "service": "Bottoms Slurry", "status": "ACTIVE"},
}

# Refinery Process Unit and Stream Label Patterns (for process flow diagrams without equipment IDs)
REFINERY_PROCESS_PATTERNS = [
    (r"\b(crude(?:\s*oil|\s*feed)?)\b", "Crude"),
    (r"\b(furnace|charge\s*heater)\b", "Furnace"),
    (r"\b(atmos(?:pheric)?\.?\s*dist(?:illation)?\.?)\b", "Atmospheric Dist."),
    (r"\b(vac(?:uum)?\.?\s*dist(?:illation)?\.?)\b", "Vacuum Dist."),
    (r"\b(fluid\s+cat(?:alytic)?\.?\s*crack(?:ing)?\.?|pfccu?|fccu?)\b", "Fluid Cat. Cracking"),
    (r"\b(vis-?breaking|visbreaker)\b", "Vis-breaking"),
    (r"\b(hydrocracker|hydrocracking|hcu)\b", "Hydrocracker"),
    (r"\b(desalter)\b", "Desalter"),
    (r"\b(gasoline)\b", "Gasoline"),
    (r"\b(kerosene(?:\s*\(jet\s*fuel\))?)\b", "Kerosene (Jet Fuel)"),
    (r"\b(gas\s*oil)\b", "Gas Oil"),
    (r"\b(heavy\s*gas\s*oil)\b", "Heavy Gas Oil"),
    (r"\b(heavy\s*resid(?:ue|\.)?|resid(?:ue|\.)?)\b", "Heavy Resid."),
    (r"\b(slurry\s*oil)\b", "Slurry Oil"),
    (r"\b(cutter\s*stock(?:\s*\(cycle\s*oil\))?)\b", "Cutter Stock (Cycle Oil)"),
    (r"\b(medium\s*fuel\s*oil)\b", "Medium fuel oil"),
    (r"\b(gases?|off-?gas)\b", "Gases"),
]


@dataclass
class VisionVerificationReport:
    """Report from vision verification pipeline.

    status values:
        VERIFIED_APPROVED  — image analyzed, all tags matched register
        VERIFIED_NO_TAGS   — image analyzed successfully, no equipment tags visible
        VERIFIED_TAGS_FOUND — alias for VERIFIED_APPROVED
        REQUIRES_REVIEW    — image analyzed, some tags unregistered
        VISION_ANALYSIS_FAILED — vision model inference failed
        NO_VERIFIABLE_TAGS — legacy alias for VERIFIED_NO_TAGS (backward compat)
    """
    status: str
    drawing_name: str
    extracted_tags: list[str]
    matched_tags: list[dict]
    unregistered_tags: list[str]
    human_approval_id: int | None = None
    process_labels: list[str] = field(default_factory=list)
    summary: str = ""
    vision_status: str = ""  # VISION_ANALYSIS_COMPLETED / VISION_ANALYSIS_FAILED / VISION_OCR_FALLBACK
    vision_error: str = ""   # Original error message if vision failed
    evidence_source: str = ""  # "vision_model", "ocr", "combined", "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "drawing_name": self.drawing_name,
            "extracted_tags": self.extracted_tags,
            "matched_tags": self.matched_tags,
            "unregistered_tags": self.unregistered_tags,
            "human_approval_id": self.human_approval_id,
            "process_labels": self.process_labels,
            "summary": self.summary,
            "vision_status": self.vision_status,
            "vision_error": self.vision_error,
            "evidence_source": self.evidence_source,
        }


@dataclass
class EquipmentTagValidationResult:
    is_approved: bool
    matched_tags: list[str] = field(default_factory=list)
    mismatched_tags: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def extract_equipment_tags_from_text(text: str) -> list[str]:
    """
    Deterministic regex extraction of refinery equipment tag patterns.
    Matches tags like: 11-P-101A, 21-R-101, 31-C-101, 11-V-101, 11-E-101A, 31-K-101.
    """
    pattern = r"\b\d{2}-[A-Z]{1,3}-\d{3}[A-Z]?\b"
    matches = re.findall(pattern, text)
    return sorted(list(set(matches)))


def extract_process_labels(text: str) -> list[str]:
    """
    Extract recognized refinery process unit and stream labels from visible text.
    Maintains appearance order while preventing duplicate canonical entries.
    """
    found = []
    seen = set()
    text_lower = text.lower()
    for regex_pattern, canonical in REFINERY_PROCESS_PATTERNS:
        if re.search(regex_pattern, text_lower, re.IGNORECASE):
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
    return found


def is_tag_grounded_in_visual_evidence(tag: str, visual_evidence: str) -> bool:
    """
    Deterministic check: Verify whether a candidate equipment tag actually
    appears in the extracted visual text evidence.
    Prevents hallucinating tags from domain knowledge or registries.
    """
    if not tag or not visual_evidence:
        return False

    clean_tag = tag.upper().strip()
    evidence_upper = visual_evidence.upper()

    if clean_tag in evidence_upper:
        return True

    # Check normalized alphanumeric match (ignoring dashes and whitespace)
    tag_chars = re.sub(r"[^A-Z0-9]", "", clean_tag)
    evidence_chars = re.sub(r"[^A-Z0-9]", "", evidence_upper)

    if len(tag_chars) >= 5 and tag_chars in evidence_chars:
        return True

    return False


def extract_grounded_equipment_tags(visual_evidence: str, proposed_text: str = "") -> list[str]:
    """
    Extracts only those equipment tags that have verifiable evidence in the image's visual text.
    Any proposed tag not supported by visual text evidence is discarded as a hallucination.
    """
    grounded = set()

    # 1. Tags directly found in extracted visual text evidence
    direct_tags = extract_equipment_tags_from_text(visual_evidence)
    for tag in direct_tags:
        grounded.add(tag)

    # 2. Candidate tags from proposed freeform model output, filtered strictly by evidence
    if proposed_text:
        proposed_candidates = extract_equipment_tags_from_text(proposed_text)
        for candidate in proposed_candidates:
            if is_tag_grounded_in_visual_evidence(candidate, visual_evidence):
                grounded.add(candidate)

    return sorted(list(grounded))


def extract_pid_tags_from_text(text: str) -> dict[str, Any]:
    """
    Extracts equipment tags and returns a structured dictionary.
    """
    tags = extract_equipment_tags_from_text(text)
    return {
        "equipment_tags": tags,
        "count": len(tags),
    }


def validate_equipment_tags(tags: list[str]) -> EquipmentTagValidationResult:
    """
    Deterministically validate a list of equipment tags against the registered MRPL equipment list.
    If no tags are present, returns is_approved=True with requires_human_review=False.
    """
    if not tags:
        return EquipmentTagValidationResult(
            is_approved=True,
            matched_tags=[],
            mismatched_tags=[],
            requires_human_review=False,
            details={},
        )

    matched = []
    mismatched = []
    details = {}

    for tag in tags:
        clean = tag.upper().strip()
        if clean in KNOWN_EQUIPMENT_REGISTER:
            matched.append(clean)
            details[clean] = KNOWN_EQUIPMENT_REGISTER[clean]
        else:
            mismatched.append(clean)

    is_approved = len(mismatched) == 0 and len(matched) > 0
    requires_human_review = len(mismatched) > 0

    return EquipmentTagValidationResult(
        is_approved=is_approved,
        matched_tags=matched,
        mismatched_tags=mismatched,
        requires_human_review=requires_human_review,
        details=details,
    )


def extract_visual_text_evidence(
    image_input: Path | str | bytes,
    model_endpoint: str = "http://127.0.0.1:11434/api/generate",
    model_name: str | None = None,
) -> str:
    """
    Extract raw visible text evidence from an image using local on-premise vision capabilities.
    First attempts PaddleOCR if available, then uses the local multimodal vision model
    in optical transcription mode.

    Returns a plain string for backward compatibility.
    For structured result with success/failure info, use extract_visual_text_evidence_v2().
    """
    result = extract_visual_text_evidence_v2(image_input, model_endpoint, model_name)
    # Return combined text for backward compatibility
    parts = []
    if result.ocr_text:
        parts.append(result.ocr_text)
    if result.vision_model_text:
        parts.append(result.vision_model_text)
    if parts:
        return "\n".join(parts)
    return result.text


def extract_visual_text_evidence_v2(
    image_input: Path | str | bytes,
    model_endpoint: str = "http://127.0.0.1:11434/api/generate",
    model_name: str | None = None,
    analysis_id: str = "",
    image_hash: str = "",
) -> VisionInferenceResult:
    """
    Extract raw visible text evidence from an image with structured success/failure tracking.

    Pipeline:
      1. Try PaddleOCR (if available)
      2. Try vision model optical transcription
      3. On vision failure: attempt ONE retry with shorter prompt and stricter config
      4. If vision fails entirely but OCR succeeded: return OCR-derived result
      5. If both fail: return explicit failure result

    Returns VisionInferenceResult with explicit success/failure state.
    """
    ocr_text = ""
    vision_text = ""
    vision_failed = False
    vision_error = ""
    vision_error_code = ""
    retried = False

    # Prepare raw bytes
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return VisionInferenceResult(
                success=False,
                error="Image file does not exist",
                error_code="FILE_NOT_FOUND",
                analysis_id=analysis_id,
                image_hash=image_hash,
            )
        with open(p, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = image_input

    # 1. Try OCR if available
    try:
        from backend.services.ocr import extract_text_from_image
        if isinstance(image_input, (str, Path)) and Path(image_input).exists():
            ocr_res = extract_text_from_image(str(image_input))
            ocr_text = ocr_res.get("text", "").strip()
    except Exception as e:
        logger.debug("OCR extraction skipped or unavailable: %s", e)

    # 2. Local Multimodal Vision Optical Transcription
    if not model_name:
        try:
            from backend.services.task_router import select_vision_model
            m_info = select_vision_model()
            model_name = m_info.get("ollama_model_name", "qwen2.5vl:3b")
            model_endpoint = m_info.get("endpoint", model_endpoint)
        except Exception:
            model_name = "qwen2.5vl:3b"

    transcription_prompt = (
        "You are an optical text transcriber. Transcribe all visible text labels, "
        "titles, unit names, and annotations printed in this image exactly as they appear. "
        "Do not invent equipment IDs. Do not analyze. Output only the exact visible words."
    )

    try:
        img_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        record_local_call(model_endpoint)
        resp = requests.post(
            model_endpoint,
            json={
                "model": model_name,
                "prompt": transcription_prompt,
                "images": [img_b64],
                "stream": False,
                "options": VISION_OCR_TRANSCRIPTION_OPTIONS,
            },
            timeout=45,
        )

        resp_text = ""
        if resp.status_code == 200:
            resp_text = resp.json().get("response", "").strip()
        else:
            resp_text = resp.text

        if is_vision_failure_response(resp.status_code, resp_text):
            # First attempt failed — try ONE retry with stricter settings
            vision_error = resp_text or f"HTTP {resp.status_code}"
            vision_error_code = _classify_vision_error(resp.status_code, resp_text)
            logger.warning(
                "Vision optical transcription failed (attempt 1): %s — %s",
                vision_error_code, vision_error[:200],
            )

            # Retry with shorter prompt and stricter config
            retried = True
            retry_prompt = "List all visible text in this image. Be brief."
            try:
                record_local_call(model_endpoint)
                retry_resp = requests.post(
                    model_endpoint,
                    json={
                        "model": model_name,
                        "prompt": retry_prompt,
                        "images": [img_b64],
                        "stream": False,
                        "options": VISION_RETRY_OPTIONS,
                    },
                    timeout=30,
                )

                retry_text = ""
                if retry_resp.status_code == 200:
                    retry_text = retry_resp.json().get("response", "").strip()
                else:
                    retry_text = retry_resp.text

                if is_vision_failure_response(retry_resp.status_code, retry_text):
                    # Retry also failed
                    vision_failed = True
                    vision_error = f"Retry also failed: {retry_text or f'HTTP {retry_resp.status_code}'}"
                    vision_error_code = _classify_vision_error(retry_resp.status_code, retry_text)
                    logger.warning("Vision optical transcription retry also failed: %s", vision_error[:200])
                else:
                    # Retry succeeded
                    vision_text = retry_text
                    vision_failed = False
                    logger.info("Vision optical transcription succeeded on retry")

            except Exception as retry_exc:
                vision_failed = True
                vision_error = f"Retry exception: {retry_exc}"
                vision_error_code = "RETRY_EXCEPTION"
                logger.warning("Vision retry exception: %s", retry_exc)
        else:
            # First attempt succeeded
            vision_text = resp_text
            vision_failed = False

    except Exception as exc:
        vision_failed = True
        vision_error = str(exc)
        vision_error_code = "CONNECTION_ERROR"
        logger.warning("Optical text evidence extraction via vision model failed: %s", exc)

    # 3. Determine final result
    combined_parts = []
    if ocr_text:
        combined_parts.append(ocr_text)
    if vision_text:
        combined_parts.append(vision_text)

    if not vision_failed and vision_text:
        # Vision succeeded (possibly combined with OCR)
        source = "combined" if ocr_text else "vision_model"
        return VisionInferenceResult(
            success=True,
            text="\n".join(combined_parts),
            source=source,
            retried=retried,
            analysis_id=analysis_id,
            image_hash=image_hash,
            ocr_text=ocr_text,
            vision_model_text=vision_text,
        )
    elif vision_failed and ocr_text:
        # Vision failed but OCR has content — fallback
        return VisionInferenceResult(
            success=True,
            text=ocr_text,
            source="ocr",
            retried=retried,
            analysis_id=analysis_id,
            image_hash=image_hash,
            ocr_text=ocr_text,
            vision_model_text="",
            error=vision_error,
            error_code=vision_error_code,
        )
    elif vision_failed:
        # Both vision and OCR failed
        return VisionInferenceResult(
            success=False,
            text="",
            error=vision_error,
            error_code=vision_error_code,
            source="none",
            retried=retried,
            analysis_id=analysis_id,
            image_hash=image_hash,
        )
    else:
        # Vision returned empty but no error — use OCR if available
        if ocr_text:
            return VisionInferenceResult(
                success=True,
                text=ocr_text,
                source="ocr",
                retried=retried,
                analysis_id=analysis_id,
                image_hash=image_hash,
                ocr_text=ocr_text,
            )
        return VisionInferenceResult(
            success=True,
            text="",
            source="vision_model",
            retried=retried,
            analysis_id=analysis_id,
            image_hash=image_hash,
        )


def build_grounded_vision_prompt(
    user_query: str,
    visual_evidence: str = "",
    process_labels: list[str] | None = None,
    grounded_tags: list[str] | None = None,
) -> str:
    """
    Constructs a zero-hallucination vision analysis prompt with mandatory 5-section structure.
    Strictly forbids inventing equipment tags, claiming MRPL attribution, or asserting compliance.
    Supports both P&ID schematics and technical software / systems diagrams factually.
    """
    labels_hint = ", ".join(process_labels) if process_labels else "None detected"
    tags_hint = ", ".join(grounded_tags) if grounded_tags else "None visible"

    return (
        "You are an objective technical vision analysis system analyzing an engineering schematic, system architecture, or process diagram.\n"
        "STRICT GROUNDING MANDATE:\n"
        "1. GROUNDING MANDATE: Only describe what is visibly depicted in the image. Do NOT invent labels, components, or IDs.\n"
        "2. NO INVENTED ATTRIBUTION: Do not claim or assume this diagram belongs to MRPL or any specific plant unless an official plant logo or ownership title block is visibly legible.\n"
        "3. NO SPECULATIVE COMPLIANCE: Do not claim engineering compliance, code adherence, or safety certification from the diagram alone.\n"
        "4. LABELS AND IDENTIFIERS: Distinguish between general descriptive block labels and specific alphanumeric equipment/component IDs.\n\n"
        f"EXTRACTED VISUAL EVIDENCE FROM IMAGE:\n"
        f"- Visible Labels Detected: {labels_hint}\n"
        f"- Verifiable Equipment/Component Tags Visible: {tags_hint}\n\n"
        f"User Inquiry: {user_query}\n\n"
        "Structure your response strictly into the following 5 numbered sections:\n"
        "### 1. Visually Observed Elements\n"
        "Describe only the visibly rendered shapes, modules, units, flow arrows, text labels, and connection streams present in the image.\n\n"
        "### 2. Process Interpretation\n"
        "Explain the high-level system architecture, process stages, data flow, or conversion steps depicted, based strictly on visible labels and connection pathways.\n\n"
        "### 3. Verifiable Equipment Tags\n"
        "List only verified alphanumeric equipment tags or component IDs that are physically legible in the image. If none are visible, explicitly state: 'No verifiable equipment tags detected. The diagram contains functional block/module labels rather than equipment IDs.'\n\n"
        "### 4. Registry Verification\n"
        "State whether equipment tags could be verified against an asset database. If the diagram depicts general technical/software architecture or has no refinery tags, explicitly state: 'No equipment registry verification performed (general technical schematic / no refinery equipment tags present).'\n\n"
        "### 5. Uncertainty / Review Required\n"
        "Identify any ambiguities, low-resolution regions, unverified assumptions, unreadable labels, or missing connection details.\n"
    )


def run_vision_analysis(
    image_bytes: bytes,
    user_query: str,
    model_endpoint: str = "http://127.0.0.1:11434/api/generate",
    model_name: str = "qwen2.5vl:3b",
    visual_evidence: str = "",
    process_labels: list[str] | None = None,
    grounded_tags: list[str] | None = None,
    analysis_id: str = "",
    image_hash: str = "",
) -> VisionInferenceResult:
    """Run the full vision analysis inference call.

    This is the SINGLE OWNER of the main Ollama vision analysis call.
    chat.py must call this function instead of making its own requests.post().

    Pipeline:
      1. Send grounded 5-section prompt to vision model with VISION_GENERATION_OPTIONS
      2. On failure: retry ONCE with build_short_vision_prompt() and VISION_RETRY_OPTIONS
      3. Return structured VisionInferenceResult

    Args:
        image_bytes: Raw image bytes (already read from file)
        user_query: The user's original query
        model_endpoint: Ollama API endpoint
        model_name: Ollama model name
        visual_evidence: Pre-extracted visual text evidence for the grounded prompt
        process_labels: Pre-extracted process labels
        grounded_tags: Pre-extracted grounded equipment tags
        analysis_id: Unique analysis identifier for tracing
        image_hash: SHA-256 hash of current image bytes
    """
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Build the full grounded prompt
    system_prompt = build_grounded_vision_prompt(
        user_query=user_query,
        visual_evidence=visual_evidence,
        process_labels=process_labels,
        grounded_tags=grounded_tags,
    )

    # Attempt 1: Full grounded analysis
    try:
        record_local_call(model_endpoint)
        resp = requests.post(
            model_endpoint,
            json={
                "model": model_name,
                "prompt": system_prompt,
                "images": [img_b64],
                "stream": False,
                "options": VISION_GENERATION_OPTIONS,
            },
            timeout=60,
        )

        resp_text = ""
        if resp.status_code == 200:
            resp_text = resp.json().get("response", "").strip()
        else:
            resp_text = resp.text

        if not is_vision_failure_response(resp.status_code, resp_text):
            # Success on first attempt
            return VisionInferenceResult(
                success=True,
                text=resp_text,
                source="vision_model",
                retried=False,
                analysis_id=analysis_id,
                image_hash=image_hash,
                vision_model_text=resp_text,
            )

        # First attempt failed — classify error
        first_error = resp_text or f"HTTP {resp.status_code}"
        first_error_code = _classify_vision_error(resp.status_code, resp_text)
        logger.warning(
            "Vision analysis failed (attempt 1): %s — %s",
            first_error_code, first_error[:200],
        )

    except requests.exceptions.Timeout:
        first_error = "Request timed out after 60s"
        first_error_code = "TIMEOUT"
        logger.warning("Vision analysis timed out (attempt 1)")
    except Exception as exc:
        first_error = str(exc)
        first_error_code = "CONNECTION_ERROR"
        logger.warning("Vision analysis exception (attempt 1): %s", exc)

    # Attempt 2: Retry with shorter prompt and stricter options
    retry_prompt = build_short_vision_prompt(user_query)
    try:
        record_local_call(model_endpoint)
        retry_resp = requests.post(
            model_endpoint,
            json={
                "model": model_name,
                "prompt": retry_prompt,
                "images": [img_b64],
                "stream": False,
                "options": VISION_RETRY_OPTIONS,
            },
            timeout=45,
        )

        retry_text = ""
        if retry_resp.status_code == 200:
            retry_text = retry_resp.json().get("response", "").strip()
        else:
            retry_text = retry_resp.text

        if not is_vision_failure_response(retry_resp.status_code, retry_text):
            # Retry succeeded
            logger.info("Vision analysis succeeded on retry")
            return VisionInferenceResult(
                success=True,
                text=retry_text,
                source="vision_model",
                retried=True,
                analysis_id=analysis_id,
                image_hash=image_hash,
                vision_model_text=retry_text,
            )

        # Retry also failed
        retry_error = retry_text or f"HTTP {retry_resp.status_code}"
        retry_error_code = _classify_vision_error(retry_resp.status_code, retry_text)
        logger.warning("Vision analysis retry also failed: %s — %s", retry_error_code, retry_error[:200])

        return VisionInferenceResult(
            success=False,
            error=f"Initial: {first_error}; Retry: {retry_error}",
            error_code=first_error_code,
            source="none",
            retried=True,
            analysis_id=analysis_id,
            image_hash=image_hash,
        )

    except Exception as retry_exc:
        logger.warning("Vision analysis retry exception: %s", retry_exc)
        return VisionInferenceResult(
            success=False,
            error=f"Initial: {first_error}; Retry exception: {retry_exc}",
            error_code=first_error_code,
            source="none",
            retried=True,
            analysis_id=analysis_id,
            image_hash=image_hash,
        )


def verify_pid_drawing(
    image_path: Path | str,
    user: dict | None = None,
    extracted_text_hint: str | None = None,
    vision_model: str | None = None,
) -> VisionVerificationReport:
    """
    Perform local multimodal vision inspection of P&ID drawing and deterministically
    validate all extracted equipment tags against the MRPL registered asset database.
    """
    path = Path(image_path)
    drawing_name = path.name

    # Step 1: Extract visual text evidence
    visual_evidence = ""
    if path.exists() and path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
        visual_evidence = extract_visual_text_evidence(path, model_name=vision_model)

    if extracted_text_hint:
        visual_evidence = f"{visual_evidence}\n{extracted_text_hint}".strip()

    # Step 2: Extract grounded equipment tags supported by visible text evidence
    extracted_tags = extract_grounded_equipment_tags(visual_evidence)
    process_labels = extract_process_labels(visual_evidence)

    user_id = user.get("id", 1) if user else 1
    username = user.get("username", "system") if user else "system"

    # If NO explicit equipment tags are visible
    if not extracted_tags:
        label_summary = f" (detected process labels: {', '.join(process_labels[:6])})" if process_labels else ""
        return VisionVerificationReport(
            status="NO_VERIFIABLE_TAGS",
            drawing_name=drawing_name,
            extracted_tags=[],
            matched_tags=[],
            unregistered_tags=[],
            human_approval_id=None,
            process_labels=process_labels,
            summary=(
                f"No verifiable equipment tags detected{label_summary}. "
                "The drawing contains process-unit and stream labels rather than specific alphanumeric equipment tags. "
                "No equipment registry verification performed."
            ),
        )

    # Step 3: Equipment registry lookup ONLY for verified visible tags
    matched = []
    unregistered = []

    for tag in extracted_tags:
        clean_tag = tag.upper().strip()
        if clean_tag in KNOWN_EQUIPMENT_REGISTER:
            info = dict(KNOWN_EQUIPMENT_REGISTER[clean_tag])
            info["tag"] = clean_tag
            matched.append(info)
        else:
            unregistered.append(clean_tag)

    if unregistered:
        status = "REQUIRES REVIEW"
        reason = f"P&ID contains {len(unregistered)} unregistered/unmatched equipment tag(s): {', '.join(unregistered)}."

        # Create approval proposal in database
        approval_id = None
        try:
            prop = approvals_repo.propose_action(
                requesting_user_id=user_id,
                requesting_username=username,
                action_type="PID_DRAWING_VERIFICATION",
                affected_resource=f"drawing:{drawing_name}",
                proposed_payload={
                    "drawing_name": drawing_name,
                    "matched_tags": matched,
                    "unregistered_tags": unregistered,
                    "reason": reason,
                },
            )
            approval_id = prop.get("approval_id")
        except Exception as e:
            logger.warning("Could not persist approval record: %s", e)

        audit_log(
            action="VISION_VERIFICATION_ESCALATED",
            outcome="denied",
            user_id=user_id,
            username=username,
            target=f"drawing:{drawing_name}",
            details={"unregistered": unregistered, "matched_count": len(matched)},
        )

        return VisionVerificationReport(
            status=status,
            drawing_name=drawing_name,
            extracted_tags=extracted_tags,
            matched_tags=matched,
            unregistered_tags=unregistered,
            human_approval_id=approval_id,
            process_labels=process_labels,
            summary=f"REQUIRES REVIEW: {reason} Tag verification escalated to plant engineering reviewer.",
        )

    # All tags matched
    audit_log(
        action="VISION_VERIFICATION_APPROVED",
        outcome="success",
        user_id=user_id,
        username=username,
        target=f"drawing:{drawing_name}",
        details={"matched": [m["tag"] for m in matched]},
    )

    return VisionVerificationReport(
        status="VERIFIED_APPROVED",
        drawing_name=drawing_name,
        extracted_tags=extracted_tags,
        matched_tags=matched,
        unregistered_tags=[],
        human_approval_id=None,
        process_labels=process_labels,
        summary=f"VERIFIED: All {len(matched)} equipment tags verified against MRPL active asset register.",
    )
