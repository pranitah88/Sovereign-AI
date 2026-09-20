"""
MRPL Sovereign AI Workbench — Unit-Aware Grounding & Cross-Unit Product Isolation.

Ensures that multi-unit technical documents (such as overview documents describing
both PFCCU and Hydrocracker in the same chunk) do not lead to cross-unit hallucination
or product/feed conflation in the LLM response.

Core Functions:
1. Target Unit Detection across English, Hindi, and Marathi.
2. Unit-attributed section segmentation for multi-unit chunks.
3. Structured context construction separating primary unit evidence from other units.
4. Deterministic post-generation grounding validation.
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Unit Definitions & Conflicting Associations ──────────────────────────────

UNIT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "hydrocracker": {
        "name": "Hydrocracker",
        "aliases": [
            r"\bhydrocracker\b", r"\bhydro-cracker\b", r"\bhydrocracking\b",
            r"\bhydro-cracking\b", r"\bhcu\b", r"हाइड्रोक्रैकर", r"हायड्रोक्रॅकर",
            r"हाइड्रो-क्रैकर", r"हायड्रो-क्रॅकर",
        ],
        "known_feeds": [
            "vacuum gas oil", "vgo", "high sulphur vacuum gas oil", "short residue", "hydrogen",
        ],
        "known_products": [
            "diesel", "ultra-pure diesel", "atf", "aviation turbine fuel",
            "lighter products", "low sulphur valuable products",
        ],
        "conflicting_products": [
            "propylene", "polypropylene", "blendstock for gasoline", "gasoline blendstock",
            "gasoline", "motor spirit", "coke", "petroleum coke",
            # Multilingual variants (Hindi/Marathi Devanagari)
            "प्रॉपिलीन", "प्रोपीलीन", "पॉलीप्रॉपिलीन", "पॉलीप्रोपाइलीन", "गॅसोलीन", "गैसोलीन",
            "पेट्रोल", "मोटर स्पिरिट", "कोक", "पेट्रोलियम कोक",
        ],
        "heading_patterns": [
            r"(?:b\.)?\s*hydro[- ]?cracking",
            r"hydrocracker\s+units?:?",
            r"two\s+hydrocrackers",
        ],
    },
    "pfccu": {
        "name": "PFCCU (Petrochemical Fluidised Catalytic Cracking Unit)",
        "aliases": [
            r"\bpfccu\b", r"\bpfcc\b", r"\bpetrochemical\s+fluidi[sz]ed\s+catalytic\s+cracker\b",
            r"\bpetroleum\s+fluidi[sz]ed\s+catalytic\s+cracker\b",
            r"\bfluidi[sz]ed\s+catalytic\s+cracking\b", r"\bfluidi[sz]ed\s+catalytic\s+cracker\b",
            r"पीएफसीसीयू", r"पीएफसीसी", r"पेट्रोकेमिकल\s+कैटेलिटिक\s+क्रैकर",
        ],
        "known_feeds": [
            "vacuum gas oil", "vgo", "unconverted oil",
            "straight run low sulphur vacuum gas oil", "hydro treated heavy coker gas oil",
        ],
        "known_products": [
            "propylene", "polymer grade propylene", "polypropylene",
            "blendstock for gasoline", "gasoline", "lpg",
        ],
        "conflicting_products": [
            "ultra-pure diesel", "diesel", "atf", "aviation turbine fuel",
            # Multilingual variants (Hindi/Marathi Devanagari)
            "डिझेल", "डीजल", "एटीएफ", "विमान टर्बाइन इंधन",
        ],
        "heading_patterns": [
            r"(?:a\.)?\s*catalytic\s+cracking:?",
            r"petroleum\s+fluidised\s+catalytic\s+cracker\s+unit\s*\(pfccu\)",
            r"pfccu\s*\(petrochemical\s+fluidized\s+catalytic\s+cracking\s+unit\)",
            r"(?:petrochemical\s+)?fluidi[sz]ed\s+catalytic\s+cracking\s*(?:\(pfccu?\)|\bpfccu?\b)?\s*units?:?",
            r"(?:l\s+)?fluidized\s+catalytic\s+cracking\s*\(pfcc\)\s*unit",
            r"pfcc\s+unit\b:?",
            r"pfccu\b:?",
        ],
    },
    "cdu_vdu": {
        "name": "CDU / VDU (Crude and Vacuum Distillation Units)",
        "aliases": [
            r"\bcdu\b", r"\bvdu\b", r"\bcrude\s+distillation\b", r"\bvacuum\s+distillation\b",
            r"सीडीयू", r"वीडीयू", r"कच्चा\s+तेल\s+आसवन",
        ],
        "known_feeds": ["crude oil", "blend crudes", "high tan crudes"],
        "known_products": ["naphtha", "kerosene", "diesel", "gas oil", "vacuum gas oil", "vgo", "vacuum residue"],
        "conflicting_products": ["propylene", "polypropylene", "प्रॉपिलीन", "प्रोपीलीन"],
        "heading_patterns": [
            r"crude\s+and\s+vacuum\s+distillation\s+units?:?",
            r"atmospheric,\s+vacuum\s+distillation\s+units?:?",
        ],
    },
    "dhdt": {
        "name": "DHDT (Diesel Hydrotreating Unit)",
        "aliases": [
            r"\bdhdt\b", r"\bdiesel\s+hydrotreating\b", r"\bdiesel\s+hydrotreater\b",
            r"\bdiesel\s+hydro\s+desulphurization\b",
            r"डीएचडीटी", r"डीजल\s+हाइड्रोसल्फराइजेशन",
        ],
        "known_feeds": ["diesel", "coker gas oil", "heavy gas oil"],
        "known_products": ["ultra-low sulphur diesel", "low sulphur diesel"],
        "conflicting_products": ["propylene", "gasoline blendstock", "प्रॉपिलीन"],
        "heading_patterns": [
            r"(?:c\.)?\s*hydro[- ]?treating",
            r"diesel\s+hydrotreating\s+units?:?",
            r"dhdt\s*\(diesel\s+hydro\s+desulphurization\s+unit\):?",
        ],
    },
    "ccr": {
        "name": "CCR Platforming (Continuous Catalytic Regeneration)",
        "aliases": [
            r"\bccr\b", r"\bplatforming\b", r"\bcontinuous\s+catalytic\s+regeneration\b",
            r"सीसीआर",
        ],
        "known_feeds": ["naphtha", "lighter naphtha"],
        "known_products": ["high octane motor spirit", "petrol", "gasoline", "hydrogen", "lpg", "reformate"],
        "conflicting_products": ["diesel", "atf", "propylene", "डिझेल", "डीजल"],
        "heading_patterns": [
            r"platforming\s+units?\s*\(ccr\):?",
            r"(?:d\.)?\s*reforming\s+and\s+isomerisation:?",
        ],
    },
    "dcu": {
        "name": "DCU (Delayed Coker Unit)",
        "aliases": [
            r"\bdcu\b", r"\bdelayed\s+coker\b", r"डीसीयू",
        ],
        "known_feeds": ["vacuum residue", "short residue"],
        "known_products": ["coke", "premium grade coke", "distillate products", "heavy gas oil"],
        "conflicting_products": ["atf"],
        "heading_patterns": [
            r"dcu\s*\(delayed\s+coker\s+unit\):?",
            r"(?:e\.)?\s*thermal\s+cracking",
        ],
    },
    "visbreaker": {
        "name": "Visbreaker Unit",
        "aliases": [
            r"\bvisbreaker\b", r"\bvisbreakers\b", r"\bvisbreaking\b",
        ],
        "known_feeds": ["heavy vacuum residue"],
        "known_products": ["gas", "naphtha", "gasoil", "vacuum gas oil"],
        "conflicting_products": ["propylene"],
        "heading_patterns": [
            r"visbreakers?\s+units?:?",
        ],
    },
    "pp_unit": {
        "name": "Polypropylene Unit (PP)",
        "aliases": [
            r"\bpp\s+unit\b", r"\bpolypropylene\s+unit\b", r"\bmangpol\b",
        ],
        "known_feeds": ["propylene", "polymer grade propylene"],
        "known_products": ["polypropylene", "polypropylene pellets", "mangpol"],
        "conflicting_products": ["diesel", "atf", "gasoline"],
        "heading_patterns": [
            r"pp\s+unit\s*\(polypropylene\s+unit\):?",
            r"polypropylene\s+unit:?",
        ],
    },
    "merox": {
        "name": "MEROX Unit",
        "aliases": [r"\bmerox\b"],
        "known_feeds": ["lpg", "kerosene", "atf"],
        "known_products": ["sweetened lpg", "sweetened kerosene", "atf"],
        "conflicting_products": ["propylene"],
        "heading_patterns": [
            r"merox\s*\(lpg\s*&\s*kerosene/atf\):?",
        ],
    },
    "hgu": {
        "name": "Hydrogen Generation Unit (HGU)",
        "aliases": [r"\bhydrogen\s+generation\b", r"\bhgu\b"],
        "known_feeds": ["naphtha"],
        "known_products": ["hydrogen"],
        "conflicting_products": ["diesel", "atf", "propylene"],
        "heading_patterns": [
            r"hydrogen\s+generation\s+units?:?",
        ],
    },
    "sru": {
        "name": "Sulphur Recovery Unit (SRU)",
        "aliases": [r"\bsru\b", r"\bsulphur\s+recovery\b"],
        "known_feeds": ["acid gas", "sour gas"],
        "known_products": ["sulphur"],
        "conflicting_products": ["diesel", "atf", "propylene"],
        "heading_patterns": [
            r"sru\s*\(sulphur\s+recovery\s+unit\):?",
        ],
    },
}


def detect_target_units(query: str) -> list[str]:
    """
    Detect all refinery units specifically mentioned in the query.
    Returns canonical keys in order of appearance: e.g. ['hydrocracker', 'pfccu'].
    """
    if not query:
        return []
    q_lower = query.lower()
    found: list[str] = []
    for unit_key, defn in UNIT_DEFINITIONS.items():
        for pat in defn["aliases"]:
            if re.search(pat, q_lower):
                if unit_key not in found:
                    found.append(unit_key)
                break
    return found


def detect_target_unit(query: str) -> Optional[str]:
    """
    Detect the primary refinery unit targeted by the query.
    Returns canonical key: 'hydrocracker', 'pfccu', 'cdu_vdu', 'dhdt', 'ccr', 'dcu', etc.
    """
    units = detect_target_units(query)
    return units[0] if units else None


def is_comparison_query(query: str) -> bool:
    """
    Check if the user query is explicitly asking for a comparison between refinery units.
    """
    if not query:
        return False
    units = detect_target_units(query)
    if len(units) >= 2:
        return True
    q_lower = query.lower()
    comp_patterns = [
        r"\bdiffer\b", r"\bdifference\b", r"\bcompare\b", r"\bcomparison\b",
        r"\bversus\b", r"\bvs\.?\b", r"तुलना", r"फरक", r"भेद", r"अंतर",
    ]
    return any(re.search(p, q_lower) for p in comp_patterns) and len(units) >= 1


def segment_text_into_unit_blocks(text: str) -> list[tuple[Optional[str], str]]:
    """
    Segment a technical text into blocks attributed to specific units.
    Uses heading patterns and unit markers.
    Returns list of (unit_key_or_None, snippet).
    If a section cannot confidently be attributed to a unit, unit_key is None (Unattributed).
    """
    if not text:
        return []

    # Compile regex pattern of all known unit headings
    heading_regexes: list[tuple[str, re.Pattern]] = []
    for unit_key, defn in UNIT_DEFINITIONS.items():
        for pat_str in defn.get("heading_patterns", []):
            heading_regexes.append((unit_key, re.compile(r'(?i)(?:^|\n|\b)(' + pat_str + r')')))

    # Break text by newlines and inline section headings
    raw_segments = re.split(r'\n{2,}|\n(?=[a-eA-E]\.|\w+[\s\w]*:)', text)
    paragraphs: list[str] = []
    for seg in raw_segments:
        sub_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z][a-zA-Z0-9\-_ /()]*:|[a-eA-E]\.)', seg)
        paragraphs.extend(sub_parts)

    blocks: list[tuple[Optional[str], str]] = []
    current_unit: Optional[str] = None

    for para in paragraphs:
        p_clean = para.strip()
        if not p_clean:
            continue

        # Check if this paragraph introduces a new unit heading
        matched_unit = None
        for u_key, regex in heading_regexes:
            if regex.search(p_clean[:140]):
                matched_unit = u_key
                break

        if matched_unit:
            current_unit = matched_unit
        else:
            # Check if paragraph explicitly mentions a single unit prominently
            mentioned_units = [
                u_key for u_key, defn in UNIT_DEFINITIONS.items()
                if any(re.search(a, p_clean.lower()) for a in defn["aliases"])
            ]
            if len(mentioned_units) == 1:
                current_unit = mentioned_units[0]
            elif len(mentioned_units) == 0 and current_unit is not None:
                # Continuation paragraph: check if it strays into generic content
                if any(kw in p_clean.lower() for kw in ["objective", "overview", "board's report", "financial"]):
                    current_unit = None

        blocks.append((current_unit, p_clean))

    return blocks


def structure_context_by_unit(
    raw_context: str,
    target_unit: Optional[str | list[str]],
) -> str:
    """
    Organize retrieved knowledge base context by separating primary evidence
    belonging to target_unit from other units, preventing cross-unit product leakage.
    Ensures that security XML wrappers (<untrusted_document_context>) remain properly closed.
    """
    if not target_unit or not raw_context:
        return raw_context

    if isinstance(target_unit, str):
        target_units = [target_unit]
    else:
        target_units = list(target_unit)

    valid_targets = [u for u in target_units if u in UNIT_DEFINITIONS]
    if not valid_targets:
        return raw_context

    from backend.services.security_guards import wrap_untrusted_context

    doc_blocks = re.split(r'(?=\[Document:\s+[^\]]+\])', raw_context)

    # Buckets: primary per target unit, other known units, unattributed/shared
    primary_blocks_map: dict[str, list[str]] = {u: [] for u in valid_targets}
    other_blocks: list[str] = []
    unattributed_blocks: list[str] = []

    for block in doc_blocks:
        b_clean = block.strip()
        if not b_clean:
            continue

        header_match = re.match(r'\[Document:\s*([^,]+),\s*Page\s*(\d+),\s*Class:\s*([^\]]+)\]', b_clean)
        if header_match:
            title = header_match.group(1).strip()
            page = int(header_match.group(2).strip())
            cls_name = header_match.group(3).strip()
            content = b_clean[header_match.end():].strip()
        else:
            title = "MRPL Technical Document"
            page = 1
            cls_name = "INTERNAL"
            content = b_clean

        content = re.sub(r'</?untrusted_document_context[^>]*>', '', content).strip()
        segmented = segment_text_into_unit_blocks(content)

        unit_snippets_map: dict[str, list[str]] = {u: [] for u in valid_targets}
        other_snippets: list[str] = []
        unattributed_snippets: list[str] = []

        for u_key, text_snippet in segmented:
            snip_lower = text_snippet.lower()

            if u_key in valid_targets:
                # Check for conflicting product in this specific snippet
                conflicting = UNIT_DEFINITIONS[u_key].get("conflicting_products", [])
                has_conflict = any(
                    re.search(r'\b' + re.escape(c.lower()) + r'\b', snip_lower) or c.lower() in snip_lower
                    for c in conflicting
                )
                if has_conflict:
                    other_snippets.append(text_snippet)
                else:
                    unit_snippets_map[u_key].append(text_snippet)
            elif u_key is not None:
                other_snippets.append(text_snippet)
            else:
                # u_key is None: check if target aliases appear without conflict
                matched_target = None
                for vt in valid_targets:
                    if any(re.search(a, snip_lower) for a in UNIT_DEFINITIONS[vt]["aliases"]):
                        conflicting = UNIT_DEFINITIONS[vt].get("conflicting_products", [])
                        if not any(re.search(r'\b' + re.escape(c.lower()) + r'\b', snip_lower) or c.lower() in snip_lower for c in conflicting):
                            matched_target = vt
                            break
                if matched_target:
                    unit_snippets_map[matched_target].append(text_snippet)
                else:
                    unattributed_snippets.append(text_snippet)

        for vt in valid_targets:
            if unit_snippets_map[vt]:
                comb = "\n\n".join(unit_snippets_map[vt])
                wrapped = wrap_untrusted_context(comb, title, page, cls_name)
                primary_blocks_map[vt].append(f"[Document: {title}, Page {page}, Class: {cls_name}]\n{wrapped}")

        if other_snippets:
            comb = "\n\n".join(other_snippets)
            wrapped = wrap_untrusted_context(comb, title, page, cls_name)
            other_blocks.append(f"[Document: {title}, Page {page}, Class: {cls_name}]\n{wrapped}")

        if unattributed_snippets:
            comb = "\n\n".join(unattributed_snippets)
            wrapped = wrap_untrusted_context(comb, title, page, cls_name)
            unattributed_blocks.append(f"[Document: {title}, Page {page}, Class: {cls_name}]\n{wrapped}")

    parts: list[str] = []

    # 1. Primary Evidence sections
    for vt in valid_targets:
        u_name = UNIT_DEFINITIONS[vt]["name"]
        if primary_blocks_map[vt]:
            parts.append(
                f"### PRIMARY EVIDENCE FOR TARGET UNIT ({u_name}):\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"- Answer claims about {u_name} ONLY from this PRIMARY EVIDENCE section.\n"
                f"- NEVER transfer a feed, product, property, purpose, or operating detail from another unit to {u_name}.\n"
                f"- Other-unit context may be used ONLY when the user explicitly asks for a comparison and the evidence clearly identifies that other unit.\n\n"
                + "\n\n".join(primary_blocks_map[vt])
            )

    # 2. Other Unit Context section
    if other_blocks:
        parts.append(
            f"### OTHER UNIT CONTEXT — DO NOT ATTRIBUTE:\n"
            f"WARNING: Do NOT attribute feeds, products, or metrics from this section to the target unit unless explicitly comparing units.\n\n"
            + "\n\n".join(other_blocks)
        )

    # 3. Unattributed / Shared Context section
    if unattributed_blocks:
        parts.append(
            f"### UNATTRIBUTED / SHARED CONTEXT:\n"
            f"NOTE: Background information not definitively attributed to a specific refinery unit.\n\n"
            + "\n\n".join(unattributed_blocks)
        )

    return "\n\n".join(parts) if parts else raw_context


def format_unit_structured_context(
    selected_chunks: list[dict],
    target_unit: Optional[str | list[str]],
) -> tuple[str, str]:
    """
    Format selected chunks into a 3-tier unit-structured context.
    Returns: (structured_context, primary_evidence_only)
    """
    from backend.services.security_guards import wrap_untrusted_context

    if not target_unit:
        blocks = []
        for r in selected_chunks:
            title = r.get("document_title") or r.get("source", "Document")
            cls_name = r.get("classification", "INTERNAL")
            wrapped = wrap_untrusted_context(
                content=r.get("text", ""),
                source=title,
                page=r.get("page", 1),
                classification=cls_name,
            )
            blocks.append(f"[Document: {title}, Page {r.get('page', 1)}, Class: {cls_name}]\n{wrapped}")
        comb = "\n\n".join(blocks)
        return comb, comb

    if isinstance(target_unit, str):
        target_units = [target_unit]
    else:
        target_units = list(target_unit)

    valid_targets = [u for u in target_units if u in UNIT_DEFINITIONS]
    if not valid_targets:
        blocks = []
        for r in selected_chunks:
            title = r.get("document_title") or r.get("source", "Document")
            cls_name = r.get("classification", "INTERNAL")
            wrapped = wrap_untrusted_context(
                content=r.get("text", ""),
                source=title,
                page=r.get("page", 1),
                classification=cls_name,
            )
            blocks.append(f"[Document: {title}, Page {r.get('page', 1)}, Class: {cls_name}]\n{wrapped}")
        comb = "\n\n".join(blocks)
        return comb, comb

    primary_blocks_map: dict[str, list[str]] = {u: [] for u in valid_targets}
    other_blocks: list[str] = []
    unattributed_blocks: list[str] = []

    for r in selected_chunks:
        title = r.get("document_title") or r.get("source", "Document")
        page = r.get("page", 1)
        cls_name = r.get("classification", "INTERNAL")
        text = r.get("text", "")

        segmented = segment_text_into_unit_blocks(text)
        unit_snippets_map: dict[str, list[str]] = {u: [] for u in valid_targets}
        other_snippets: list[str] = []
        unattributed_snippets: list[str] = []

        for u_key, snippet in segmented:
            snip_lower = snippet.lower()

            if u_key in valid_targets:
                conflicting = UNIT_DEFINITIONS[u_key].get("conflicting_products", [])
                has_conflict = any(
                    re.search(r'\b' + re.escape(c.lower()) + r'\b', snip_lower) or c.lower() in snip_lower
                    for c in conflicting
                )
                if has_conflict:
                    other_snippets.append(snippet)
                else:
                    unit_snippets_map[u_key].append(snippet)
            elif u_key is not None:
                other_snippets.append(snippet)
            else:
                matched_target = None
                for vt in valid_targets:
                    if any(re.search(a, snip_lower) for a in UNIT_DEFINITIONS[vt]["aliases"]):
                        conflicting = UNIT_DEFINITIONS[vt].get("conflicting_products", [])
                        if not any(re.search(r'\b' + re.escape(c.lower()) + r'\b', snip_lower) or c.lower() in snip_lower for c in conflicting):
                            matched_target = vt
                            break
                if matched_target:
                    unit_snippets_map[matched_target].append(snippet)
                else:
                    unattributed_snippets.append(snippet)

        for vt in valid_targets:
            if unit_snippets_map[vt]:
                comb = "\n\n".join(unit_snippets_map[vt])
                wrapped = wrap_untrusted_context(comb, title, page, cls_name)
                primary_blocks_map[vt].append(f"[Document: {title}, Page {page}, Class: {cls_name}]\n{wrapped}")

        if other_snippets:
            comb = "\n\n".join(other_snippets)
            wrapped = wrap_untrusted_context(comb, title, page, cls_name)
            other_blocks.append(f"[Document: {title}, Page {page}, Class: {cls_name}]\n{wrapped}")

        if unattributed_snippets:
            comb = "\n\n".join(unattributed_snippets)
            wrapped = wrap_untrusted_context(comb, title, page, cls_name)
            unattributed_blocks.append(f"[Document: {title}, Page {page}, Class: {cls_name}]\n{wrapped}")

    all_primary_blocks: list[str] = []
    for vt in valid_targets:
        all_primary_blocks.extend(primary_blocks_map[vt])
    primary_evidence_only = "\n\n".join(all_primary_blocks)

    parts: list[str] = []

    for vt in valid_targets:
        u_name = UNIT_DEFINITIONS[vt]["name"]
        if primary_blocks_map[vt]:
            parts.append(
                f"### PRIMARY EVIDENCE FOR TARGET UNIT ({u_name}):\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"- Answer claims about {u_name} ONLY from this PRIMARY EVIDENCE section.\n"
                f"- NEVER transfer a feed, product, property, purpose, or operating detail from another unit to {u_name}.\n"
                f"- Other-unit context may be used ONLY when the user explicitly asks for a comparison and the evidence clearly identifies that other unit.\n\n"
                + "\n\n".join(primary_blocks_map[vt])
            )

    if other_blocks:
        parts.append(
            f"### OTHER UNIT CONTEXT — DO NOT ATTRIBUTE:\n"
            f"WARNING: Do NOT attribute feeds, products, or metrics from this section to the target unit unless explicitly comparing units.\n\n"
            + "\n\n".join(other_blocks)
        )

    if unattributed_blocks:
        parts.append(
            f"### UNATTRIBUTED / SHARED CONTEXT:\n"
            f"NOTE: Background information not definitively attributed to a specific refinery unit.\n\n"
            + "\n\n".join(unattributed_blocks)
        )

    full_context = "\n\n".join(parts) if parts else primary_evidence_only
    return full_context, primary_evidence_only


def validate_unit_grounding(
    response: str,
    target_unit: Optional[str | list[str]],
    primary_evidence: Optional[str] = None,
    all_evidence: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Evidence-based post-generation grounding validator.
    Determines whether claims about the TARGET UNIT are supported by TARGET UNIT evidence.

    Rules:
    1. Identify factual claims in the response body (ignoring Sources / References section).
    2. Check which unit is the subject of each claim / clause.
    3. If a claim attributes feeds/products to the target unit:
       Verify that the claim is supported by primary target-unit evidence.
    4. If the response attributes a product of another unit to that other unit
       (e.g., in comparisons: "PFCCU produces propylene, while Hydrocracker produces diesel and ATF"):
       Verify both claims against their respective supporting evidence without failing.
    5. Fail when information belonging to another unit is actually attributed to the
       target unit without supporting evidence in the primary evidence.

    Returns: (is_valid, failure_reason)
    """
    if not target_unit or not response:
        return True, None

    if isinstance(target_unit, str):
        target_units = [target_unit]
    else:
        target_units = list(target_unit)

    valid_targets = [u for u in target_units if u in UNIT_DEFINITIONS]
    if not valid_targets:
        return True, None

    # 1. Strip Sources / References sections and citations so footnote titles don't trigger false alarms
    body_text = re.split(r'\n+(?:##\s*)?(?:Sources|References):\s*\n', response, flags=re.IGNORECASE)[0]
    body_text = re.sub(r'\((?:Source|Doc):\s*[^)]+\)', '', body_text, flags=re.IGNORECASE)
    body_text = re.sub(r'\[(?:Source|Doc):\s*[^\]]+\]', '', body_text, flags=re.IGNORECASE)

    # 2. Break body text into sentences and sub-clauses
    sentences = re.split(r'(?<=[.!?\n])\s+', body_text.strip())

    clauses: list[str] = []
    for sent in sentences:
        s_clean = sent.strip()
        if not s_clean:
            continue
        # Split on contrastive / comparative conjunctions within sentence
        # English: while, whereas, but, however, on the other hand
        # Marathi: तर, दरम्यान, मात्र, उलट
        # Hindi: जबकि, किन्तु, परन्तु, दूसरी ओर
        sub_clauses = re.split(
            r'(?i)(?:,\s*(?:while|whereas|but|however|on the other hand)|;\s*|,\s*(?:तर|दरम्यान|मात्र|उलट)|,\s*(?:जबकि|किन्तु|परन्तु))\s*',
            s_clean
        )
        clauses.extend([c.strip() for c in sub_clauses if c.strip()])

    # Context evidence strings for verification
    pe_lower = (primary_evidence or "").lower()
    ae_lower = (all_evidence or "").lower()

    last_subject: Optional[str] = valid_targets[0] if len(valid_targets) == 1 else None

    for clause in clauses:
        clause_lower = clause.lower()

        # Identify all units mentioned in this clause
        mentioned_in_clause: list[str] = []
        for u_k, defn in UNIT_DEFINITIONS.items():
            for alias in defn["aliases"]:
                if re.search(alias, clause_lower):
                    if u_k not in mentioned_in_clause:
                        mentioned_in_clause.append(u_k)
                    break

        if len(mentioned_in_clause) == 1:
            clause_subject = mentioned_in_clause[0]
            last_subject = clause_subject
        elif len(mentioned_in_clause) > 1:
            clause_subject = None
        else:
            # Check for pronoun or continuation
            has_pronoun_or_continuation = bool(re.search(
                r'\b(it|its|the unit|this unit|these units|या युनिटची|या युनिट|हे युनिट|इसके|इस इकाई|इस यूनिट|यह इकाई)\b',
                clause_lower
            ))
            if has_pronoun_or_continuation and last_subject:
                clause_subject = last_subject
            elif len(valid_targets) == 1:
                clause_subject = valid_targets[0]
            else:
                clause_subject = last_subject

        # Determine which units this clause makes claims about
        units_to_check = [clause_subject] if clause_subject else mentioned_in_clause
        if not units_to_check and len(valid_targets) == 1:
            units_to_check = [valid_targets[0]]

        for u_k in units_to_check:
            if not u_k or u_k not in UNIT_DEFINITIONS:
                continue

            u_defn = UNIT_DEFINITIONS[u_k]
            u_name = u_defn["name"]
            conflicting = u_defn.get("conflicting_products", [])

            for conf in conflicting:
                conf_lower = conf.lower()
                conf_pat = r'\b' + re.escape(conf_lower) + r'\b'
                has_conf = bool(re.search(conf_pat, clause_lower) or (len(conf_lower) > 3 and conf_lower in clause_lower))

                if has_conf:
                    # The clause attributes this conflicting product/term to u_k
                    # Check if evidence actually supports it for u_k
                    is_supported = False
                    if primary_evidence and u_k in valid_targets:
                        is_supported = bool(re.search(conf_pat, pe_lower) or conf_lower in pe_lower)
                    elif all_evidence:
                        is_supported = bool(re.search(conf_pat, ae_lower) or conf_lower in ae_lower)

                    if not is_supported:
                        reason = (
                            f"Grounding violation: Response attributes conflicting product/term '{conf}' "
                            f"to {u_name}, but supporting evidence for {u_name} does not contain this product/feed."
                        )
                        logger.warning(reason)
                        return False, reason

    return True, None
