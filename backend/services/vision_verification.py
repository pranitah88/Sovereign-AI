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
    status: str  # "VERIFIED_APPROVED" | "REQUIRES REVIEW" | "NO_VERIFIABLE_TAGS" | "FAILED"
    drawing_name: str
    extracted_tags: list[str]
    matched_tags: list[dict]
    unregistered_tags: list[str]
    human_approval_id: int | None = None
    process_labels: list[str] = field(default_factory=list)
    summary: str = ""

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
    in optical transcription mode (temperature=0.0).
    """
    evidence_parts = []

    # Prepare raw bytes
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return ""
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
            if ocr_text:
                evidence_parts.append(ocr_text)
    except Exception as e:
        logger.debug("OCR extraction skipped or unavailable: %s", e)

    # 2. Local Multimodal Vision Optical Transcription (deterministic temperature 0.0)
    if not model_name:
        try:
            from backend.services.task_router import select_vision_model
            m_info = select_vision_model()
            model_name = m_info.get("ollama_model_name", "qwen2.5vl:3b")
            model_endpoint = m_info.get("endpoint", model_endpoint)
        except Exception:
            model_name = "qwen2.5vl:3b"

    try:
        img_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        record_local_call(model_endpoint)
        resp = requests.post(
            model_endpoint,
            json={
                "model": model_name,
                "prompt": (
                    "You are an optical text transcriber. Transcribe all visible text labels, "
                    "titles, unit names, and annotations printed in this image exactly as they appear. "
                    "Do not invent equipment IDs. Do not analyze. Output only the exact visible words."
                ),
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 300},
            },
            timeout=45,
        )
        if resp.status_code == 200:
            vision_transcription = resp.json().get("response", "").strip()
            if vision_transcription:
                evidence_parts.append(vision_transcription)
    except Exception as exc:
        logger.warning("Optical text evidence extraction via vision model failed: %s", exc)

    return "\n".join(evidence_parts)


def build_grounded_vision_prompt(
    user_query: str,
    visual_evidence: str = "",
    process_labels: list[str] | None = None,
    grounded_tags: list[str] | None = None,
) -> str:
    """
    Constructs a zero-hallucination vision analysis prompt with mandatory 5-section structure.
    Strictly forbids inventing equipment tags, claiming MRPL attribution, or asserting compliance.
    """
    labels_hint = ", ".join(process_labels) if process_labels else "None detected"
    tags_hint = ", ".join(grounded_tags) if grounded_tags else "None visible"

    return (
        "You are an objective industrial engineering vision system analyzing an engineering schematic or diagram.\n"
        "STRICT GROUNDING MANDATE:\n"
        "1. GROUNDING MANDATE: Only describe what is visibly depicted. Do NOT invent equipment tags, serial numbers, or asset IDs.\n"
        "2. NO INVENTED ATTRIBUTION: Do not claim or assume this diagram belongs to MRPL or any specific plant unless an official plant logo or ownership title block is visibly legible.\n"
        "3. NO SPECULATIVE COMPLIANCE: Do not claim engineering compliance, code adherence, or safety certification from the diagram alone.\n"
        "4. EQUIPMENT TAG DISTINCTION: Note the difference between general process block labels (e.g. 'Furnace', 'Atmospheric Distillation', 'Gas Oil') and specific equipment tags (e.g. '11-P-101A'). Do not classify process labels as equipment tags.\n\n"
        f"EXTRACTED VISUAL EVIDENCE FROM IMAGE:\n"
        f"- Visible Process Labels Detected: {labels_hint}\n"
        f"- Verifiable Equipment Tags Visible: {tags_hint}\n\n"
        f"User Inquiry: {user_query}\n\n"
        "Structure your response strictly into the following 5 numbered sections:\n"
        "### 1. Visually Observed Elements\n"
        "Describe only the visibly rendered shapes, units, flow arrows, text labels, and streams present in the image.\n\n"
        "### 2. Process Interpretation\n"
        "Explain the high-level refining process flow depicted (feedstocks, intermediate separations, cracking, conversion, products) based strictly on visible labels.\n\n"
        "### 3. Verifiable Equipment Tags\n"
        "List only verified alphanumeric equipment tags that are physically legible in the image. If none are visible, explicitly state: 'No verifiable equipment tags detected. The diagram contains process-unit/stream labels rather than equipment IDs.'\n\n"
        "### 4. Registry Verification\n"
        "State whether equipment tags could be verified against an asset database. If no equipment tags are visible, explicitly state: 'No equipment registry verification performed (no equipment tags present).'\n\n"
        "### 5. Uncertainty / Review Required\n"
        "Identify any ambiguities, low-resolution regions, unverified assumptions, or missing instrumentation details.\n"
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
