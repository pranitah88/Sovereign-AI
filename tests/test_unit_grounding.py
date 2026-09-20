"""
Automated Unit-Aware Grounding & Cross-Unit Product Isolation Tests.

Covers all 8 specified verification tests:
TEST 1: English Hydrocracker ("What feed does the Hydrocracker process and what are its major products?")
TEST 2: Hindi Hydrocracker ("हाइड्रोक्रैकर किस फीड पर प्रक्रिया करता है और इसके प्रमुख उत्पाद क्या हैं?")
TEST 3: Marathi Hydrocracker ("हायड्रोक्रॅकर कोणत्या फीडवर प्रक्रिया करतो आणि त्याची प्रमुख उत्पादने कोणती आहेत?")
TEST 4: PFCCU ("What are the major products of PFCCU?")
TEST 5: Explicit Comparison ("How does the Hydrocracker differ from the PFCCU in terms of feed and products?")
TEST 6: Irrelevant Source Rejection (Job scam notice rejected / not ranked above Hydrocracker evidence)
TEST 7: Multi-unit Chunk Isolation (Known chunk from 'MRPL CUTTING EDGE TECHNOLOGY IN REFINING')
TEST 8: Grounding Validator (Unsupported claim rejected, supported claim accepted, valid comparison accepted)
"""

import pytest

from backend.services.unit_grounding import (
    UNIT_DEFINITIONS,
    detect_target_unit,
    detect_target_units,
    is_comparison_query,
    format_unit_structured_context,
    segment_text_into_unit_blocks,
    structure_context_by_unit,
    validate_unit_grounding,
)
from backend.services.rag_engine import query_knowledge_base


# ── Target Unit Detection across EN, HI, MR ─────────────────────────────────

def test_detect_target_unit_hydrocracker():
    """Verify Hydrocracker detection in English, Hindi, and Marathi."""
    en = "What feed does the Hydrocracker process and what are its major products?"
    hi = "हाइड्रोक्रैकर किस फीड पर प्रक्रिया करता है और इसके प्रमुख उत्पाद क्या हैं?"
    mr = "हायड्रोक्रॅकर कोणत्या फीडवर प्रक्रिया करतो आणि त्याची प्रमुख उत्पादने कोणती आहेत?"
    mr_mrpl = "MRPL च्या हायड्रोक्रॅकर युनिटमध्ये कोणत्या फीडवर प्रक्रिया केली जाते आणि या युनिटची प्रमुख उत्पादने कोणती आहेत?"

    assert detect_target_unit(en) == "hydrocracker"
    assert detect_target_unit(hi) == "hydrocracker"
    assert detect_target_unit(mr) == "hydrocracker"
    assert detect_target_unit(mr_mrpl) == "hydrocracker"


def test_detect_target_unit_pfccu():
    """Verify PFCCU detection in English, Hindi, and Marathi."""
    en = "What are the major products of PFCCU?"
    hi = "पीएफसीसीयू के प्रमुख उत्पाद क्या हैं?"
    mr = "PFCCU चे प्रमुख उत्पादने कोणती आहेत?"

    assert detect_target_unit(en) == "pfccu"
    assert detect_target_unit(hi) == "pfccu"
    assert detect_target_unit(mr) == "pfccu"


def test_detect_target_unit_other_units():
    """Verify detection for other refinery units and negative cases."""
    assert detect_target_unit("What is the role of CDU and VDU at MRPL?") == "cdu_vdu"
    assert detect_target_unit("MRPL में डीएचडीटी यूनिट क्या करती है?") == "dhdt"
    assert detect_target_unit("What does the CCR Platforming unit produce?") == "ccr"
    assert detect_target_unit("How does the Delayed Coker Unit work?") == "dcu"
    assert detect_target_unit("What is the feed for Visbreaker?") == "visbreaker"
    assert detect_target_unit("How does the Hydrogen Generation Unit operate?") == "hgu"
    assert detect_target_unit("What is the capacity of Sulphur Recovery Unit?") == "sru"
    # Negative cases
    assert detect_target_unit("What is the total revenue of MRPL in FY 2023-24?") is None
    assert detect_target_unit("Who is the Managing Director of MRPL?") is None


# ── TEST 7: Multi-Unit Chunk Isolation ──────────────────────────────────────

def test_multi_unit_chunk_isolation():
    """
    TEST 7: Use the known multi-unit chunk containing PFCCU and Hydrocracker.
    Verify:
    Hydrocracker evidence -> Hydrocracker
    PFCCU evidence -> PFCCU
    No cross-attribution.
    """
    known_multi_unit_chunk = (
        "tic Cracking: Catalytic cracking in MRPL is achieved through Petroleum Fluidised Catalytic "
        "Cracker Unit (PFCCU), wherein, the feed Vacuum Gas Oil is reacted on a moving fluidised "
        "catalytic bed. MRPL PFCCU is designed to provide a high yield of Propylene, which is the raw "
        "material for valuable Polypropylene Product. In addition to Propylene, the unit generates "
        "blendstock for Gasoline. "
        "b.Hydro-Cracking hydrogen hydrocracking MRPL is having two hydrocrackers which produce "
        "ultra-pure diesel by processing High Sulphur Vacuum Gas Oil. The unit operates at 190 kscg "
        "pressure while the feed reacts with hydrogen for the production of lighter products. "
        "c.Hydro-Treating In MRPL, there are two Diesel hydrotreating units, wherein treatment in the "
        "presence of hydrogen yields ultra-low Sulphur diesel."
    )

    blocks = segment_text_into_unit_blocks(known_multi_unit_chunk)
    assert len(blocks) >= 3

    units_found = [b[0] for b in blocks if b[0] is not None]
    assert "pfccu" in units_found
    assert "hydrocracker" in units_found
    assert "dhdt" in units_found

    # Test Hydrocracker perspective
    selected = [{
        "document_title": "MRPL CUTTING EDGE TECHNOLOGY IN REFINING",
        "page": 1,
        "classification": "PUBLIC",
        "text": known_multi_unit_chunk,
    }]
    hc_full, hc_primary = format_unit_structured_context(selected, "hydrocracker")
    assert "### PRIMARY EVIDENCE FOR TARGET UNIT (Hydrocracker):" in hc_full
    assert "ultra-pure diesel" in hc_primary
    assert "High Sulphur Vacuum Gas Oil" in hc_primary
    assert "propylene" not in hc_primary.lower()
    assert "blendstock for gasoline" not in hc_primary.lower()

    # Test PFCCU perspective
    pf_full, pf_primary = format_unit_structured_context(selected, "pfccu")
    assert "### PRIMARY EVIDENCE FOR TARGET UNIT (PFCCU" in pf_full
    assert "Propylene" in pf_primary
    assert "blendstock for Gasoline" in pf_primary
    assert "ultra-pure diesel" not in pf_primary.lower()


# ── TEST 8: Grounding Validator ─────────────────────────────────────────────

def test_grounding_validator_test8():
    """
    TEST 8 — Grounding validator
    Test 8A: "The Hydrocracker produces propylene." against Hydrocracker evidence that does not support propylene -> INVALID
    Test 8B: "The PFCCU produces propylene." against supporting PFCCU evidence -> VALID
    Test 8C: "The PFCCU produces propylene, while the Hydrocracker produces diesel and ATF." when both are supported -> VALID
    """
    hc_evidence = (
        "MRPL has two hydrocrackers which produce ultra-pure diesel and ATF by processing "
        "High Sulphur Vacuum Gas Oil at 190 kscg pressure."
    )
    pfccu_evidence = (
        "MRPL PFCCU is designed to provide a high yield of Propylene, which is the raw material "
        "for Polypropylene. In addition to Propylene, the unit generates blendstock for Gasoline."
    )
    combined_evidence = f"{pfccu_evidence}\n\n{hc_evidence}"

    # Test 8A: Hydrocracker produces propylene -> INVALID
    resp_8a = "The Hydrocracker produces propylene."
    valid_8a, reason_8a = validate_unit_grounding(
        response=resp_8a,
        target_unit="hydrocracker",
        primary_evidence=hc_evidence,
        all_evidence=hc_evidence,
    )
    assert valid_8a is False
    assert "propylene" in reason_8a.lower()

    # Test 8B: PFCCU produces propylene -> VALID
    resp_8b = "The PFCCU produces propylene."
    valid_8b, reason_8b = validate_unit_grounding(
        response=resp_8b,
        target_unit="pfccu",
        primary_evidence=pfccu_evidence,
        all_evidence=pfccu_evidence,
    )
    assert valid_8b is True
    assert reason_8b is None

    # Test 8C: Comparison where both statements are supported -> VALID
    resp_8c = "The PFCCU produces propylene, while the Hydrocracker produces diesel and ATF."
    valid_8c, reason_8c = validate_unit_grounding(
        response=resp_8c,
        target_unit=["pfccu", "hydrocracker"],
        primary_evidence=combined_evidence,
        all_evidence=combined_evidence,
    )
    assert valid_8c is True
    assert reason_8c is None

    # Also test Marathi Devanagari grounding violation
    mr_bad = "हायड्रोक्रॅकर युनिटचे मुख्य उत्पादन प्रॉपिलीन आहे."
    valid_mr, reason_mr = validate_unit_grounding(
        response=mr_bad,
        target_unit="hydrocracker",
        primary_evidence=hc_evidence,
    )
    assert valid_mr is False
    assert "प्रॉपिलीन" in reason_mr or "propylene" in reason_mr.lower()


# ── TEST 1: English Hydrocracker ────────────────────────────────────────────

def test_rag_hydrocracker_english():
    """
    TEST 1: English Hydrocracker
    Query: 'What feed does the Hydrocracker process and what are its major products?'
    Expected:
    - Hydrocracker evidence retrieved
    - High Sulphur Vacuum Gas Oil / relevant feed terminology
    - Diesel / ATF
    - Must NOT incorrectly attribute Propylene or Gasoline blendstock to Hydrocracker.
    """
    query = "What feed does the Hydrocracker process and what are its major products?"
    res = query_knowledge_base(query, top_k=3)

    assert res["refusal"] is False
    assert res["target_unit"] == "hydrocracker"
    assert len(res["sources"]) > 0

    primary_ev = res.get("primary_unit_evidence", "").lower()
    assert ("diesel" in primary_ev or "vacuum gas oil" in primary_ev or "vgo" in primary_ev or "sulphur" in primary_ev)
    assert "propylene" not in primary_ev
    assert "blendstock for gasoline" not in primary_ev


# ── TEST 2: Hindi Hydrocracker ──────────────────────────────────────────────

def test_rag_hydrocracker_hindi():
    """
    TEST 2: Hindi Hydrocracker
    Query: 'हाइड्रोक्रैकर किस फीड पर प्रक्रिया करता है और इसके प्रमुख उत्पाद क्या हैं?'
    Expected:
    - Target unit detected as hydrocracker
    - Hydrocracker technical evidence retrieved
    - No unsupported PFCCU products attributed to Hydrocracker
    """
    query = "हाइड्रोक्रैकर किस फीड पर प्रक्रिया करता है और इसके प्रमुख उत्पाद क्या हैं?"
    res = query_knowledge_base(query, top_k=3)

    assert res["refusal"] is False
    assert res["target_unit"] == "hydrocracker"
    assert len(res["sources"]) > 0

    primary_ev = res.get("primary_unit_evidence", "").lower()
    assert "propylene" not in primary_ev
    assert "blendstock for gasoline" not in primary_ev


# ── TEST 3: Marathi Hydrocracker ────────────────────────────────────────────

def test_rag_hydrocracker_marathi():
    """
    TEST 3: Marathi Hydrocracker
    Query: 'हायड्रोक्रॅकर कोणत्या फीडवर प्रक्रिया करतो आणि त्याची प्रमुख उत्पादने कोणती आहेत?'
    Expected:
    - Target unit detected as hydrocracker
    - Hydrocracker technical evidence retrieved
    - No unsupported PFCCU products attributed to Hydrocracker
    """
    query = "हायड्रोक्रॅकर कोणत्या फीडवर प्रक्रिया करतो आणि त्याची प्रमुख उत्पादने कोणती आहेत?"
    res = query_knowledge_base(query, top_k=3)

    assert res["refusal"] is False
    assert res["target_unit"] == "hydrocracker"
    assert len(res["sources"]) > 0

    primary_ev = res.get("primary_unit_evidence", "").lower()
    assert "propylene" not in primary_ev
    assert "blendstock for gasoline" not in primary_ev


# ── TEST 4: PFCCU ───────────────────────────────────────────────────────────

def test_rag_pfccu_products():
    """
    TEST 4: PFCCU
    Query: 'What are the major products of PFCCU?'
    Expected:
    - Target unit detected as pfccu
    - PFCCU technical evidence retrieved (Propylene / gasoline blendstock / polypropylene)
    - Must NOT attribute Hydrocracker-only products (ultra-pure diesel / ATF) to PFCCU
    """
    query = "What are the major products of PFCCU?"
    res = query_knowledge_base(query, top_k=3)

    assert res["refusal"] is False
    assert res["target_unit"] == "pfccu"
    assert len(res["sources"]) > 0

    primary_ev = res.get("primary_unit_evidence", "").lower()
    assert ("propylene" in primary_ev or "gasoline" in primary_ev or "polypropylene" in primary_ev)
    assert "ultra-pure diesel" not in primary_ev


# ── TEST 5: Explicit Comparison ─────────────────────────────────────────────

def test_rag_explicit_comparison_hydrocracker_vs_pfccu():
    """
    TEST 5: Explicit comparison
    Query: 'How does the Hydrocracker differ from the PFCCU in terms of feed and products?'
    Expected:
    - Both units are detected
    - Comparison query is recognized
    - Both units' primary evidence are provided
    - Cross-unit mixing is prohibited
    """
    query = "How does the Hydrocracker differ from the PFCCU in terms of feed and products?"
    units = detect_target_units(query)
    assert "hydrocracker" in units
    assert "pfccu" in units
    assert is_comparison_query(query) is True

    res = query_knowledge_base(query, top_k=3)
    assert res["refusal"] is False
    assert len(res["sources"]) > 0

    context = res.get("context", "")
    assert "PRIMARY EVIDENCE FOR TARGET UNIT (Hydrocracker):" in context
    assert "PRIMARY EVIDENCE FOR TARGET UNIT (PFCCU" in context


# ── TEST 6: Irrelevant Source Rejection ──────────────────────────────────────

def test_irrelevant_source_rejection_for_unit_query():
    """
    TEST 6: Irrelevant source rejection
    Hydrocracker query should NOT rank a job scam notice above technical Hydrocracker evidence.
    """
    query = "What is the purpose of the Hydrocracker unit at MRPL?"
    res = query_knowledge_base(query, top_k=3)

    assert res["refusal"] is False
    assert len(res["sources"]) > 0

    # Verify top sources are technical refining documents, NOT job scam or recruitment notices
    for s in res["sources"]:
        title_lower = s.get("document_title", "").lower()
        text_lower = s.get("text", "").lower()
        assert "job scam" not in title_lower
        assert "job scam" not in text_lower
        assert "सार्वजनिक सूचना" not in s.get("document_title", "")
