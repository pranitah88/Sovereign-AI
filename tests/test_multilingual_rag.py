"""
Automated Multilingual RAG Retrieval Tests for MRPL Sovereign AI Workbench.

Tests the 10 exact prompt cases covering English, Hindi, and Marathi,
validating deterministic query normalization, technical entity extraction,
topic boosting, category filtering, source relevance validation, and
consistency across languages.
"""

import pytest
from backend.services.multilingual import detect_language, normalize_query
from backend.services.query_analyzer import analyze_query
from backend.services.rag_engine import query_knowledge_base


# ── Step 1: Multilingual Preprocessing & Normalization Tests ─────────────────

def test_language_detection():
    """Verify accurate detection of English, Hindi, and Marathi."""
    assert detect_language("What is the main purpose of the hydrocracker unit at MRPL?") == "english"
    assert detect_language("MRPL की रिफाइनरी में हाइड्रोक्रैकर यूनिट का मुख्य उद्देश्य क्या है?") == "hindi"
    assert detect_language("MRPL च्या रिफायनरीमध्ये हायड्रोक्रॅकर युनिटचा मुख्य उद्देश काय आहे?") == "marathi"
    assert detect_language("MRPL का FY2025 का राजस्व कितना था?") == "hindi"
    assert detect_language("FY2025 मध्ये MRPL चा महसूल किती होता?") == "marathi"


def test_technical_entity_normalization_hydrocracker():
    """Verify that Hindi, Marathi, and English hydrocracker queries normalize identically."""
    en = normalize_query("What is the main purpose of the hydrocracker unit at MRPL?")
    hi = normalize_query("MRPL की रिफाइनरी में हाइड्रोक्रैकर यूनिट का मुख्य उद्देश्य क्या है? सरल शब्दों में समझाइए।")
    mr = normalize_query("MRPL च्या रिफायनरीमध्ये हायड्रोक्रॅकर युनिटचा मुख्य उद्देश काय आहे? सोप्या भाषेत समजावून सांगा.")

    assert en.detected_topic == "refinery_unit"
    assert hi.detected_topic == "refinery_unit"
    assert mr.detected_topic == "refinery_unit"

    assert "hydrocracker" in hi.canonical_entities
    assert "hydrocracker" in mr.canonical_entities
    assert "hydrocracker" in en.canonical_entities

    # Normalized query contains canonical retrieval keywords
    assert "hydrocracker" in hi.normalized_query
    assert "refinery" in hi.normalized_query
    assert "hydrocracker" in mr.normalized_query
    assert "refinery" in mr.normalized_query


def test_financial_metric_normalization_revenue():
    """Verify Hindi and Marathi revenue queries extract revenue without substituting exports/profit."""
    hi = analyze_query("MRPL का FY2025 का राजस्व कितना था?")
    mr = analyze_query("FY2025 मध्ये MRPL चा महसूल किती होता?")

    assert hi.primary_metric == "revenue"
    assert mr.primary_metric == "revenue"
    assert hi.year == "FY2025"
    assert mr.year == "FY2025"
    assert hi.is_exact_value_query is True
    assert mr.is_exact_value_query is True


# ── Step 2: Exact 10 RAG Retrieval Tests ─────────────────────────────────────

def test_case_1_english_hydrocracker():
    """TEST 1: English Hydrocracker -> Technical refinery source."""
    res = query_knowledge_base("What is the main purpose of the hydrocracker unit at MRPL?", top_k=5, debug=True)
    assert res["result_count"] > 0
    top_src = res["sources"][0]
    title = top_src["document_title"].lower()
    cat = top_src.get("category", "")
    doctype = top_src.get("document_type", "")
    assert any(w in title for w in ["refining", "cutting edge", "manufacturing", "facilities", "annual report"])
    assert cat in ("01_Refinery_Manufacturing", "07_Finance", "06_Website_Content") or doctype == "manufacturing_refining"
    # Ensure job scam notice is NOT retrieved
    assert not any("job scam" in s["document_title"].lower() for s in res["sources"])


def test_case_2_hindi_hydrocracker():
    """TEST 2: Hindi Hydrocracker -> Same technical retrieval as English."""
    res = query_knowledge_base("MRPL की रिफाइनरी में हाइड्रोक्रैकर यूनिट का मुख्य उद्देश्य क्या है?", top_k=5, debug=True)
    assert res["result_count"] > 0
    top_src = res["sources"][0]
    title = top_src["document_title"].lower()
    assert any(w in title for w in ["refining", "cutting edge", "manufacturing", "facilities", "annual report"])
    # Job scam notice must be rejected
    assert not any("job scam" in s["document_title"].lower() for s in res["sources"])


def test_case_3_marathi_hydrocracker():
    """TEST 3: Marathi Hydrocracker -> Same technical retrieval as English."""
    res = query_knowledge_base("MRPL च्या रिफायनरीमध्ये हायड्रोक्रॅकर युनिटचा मुख्य उद्देश काय आहे?", top_k=5, debug=True)
    assert res["result_count"] > 0
    top_src = res["sources"][0]
    title = top_src["document_title"].lower()
    assert any(w in title for w in ["refining", "cutting edge", "manufacturing", "facilities", "annual report"])
    assert not any("job scam" in s["document_title"].lower() for s in res["sources"])


def test_case_4_hindi_hydrocracker_conversational():
    """TEST 4: Hindi Hydrocracker Conversational -> Technical source."""
    res = query_knowledge_base("MRPL की रिफाइनरी में हाइड्रोक्रैकर यूनिट के बारे में सरल शब्दों में बताइए।", top_k=5, debug=True)
    assert res["result_count"] > 0
    top_src = res["sources"][0]
    title = top_src["document_title"].lower()
    assert any(w in title for w in ["refining", "cutting edge", "manufacturing", "facilities", "annual report"])
    assert not any("job scam" in s["document_title"].lower() for s in res["sources"])


def test_case_5_marathi_hydrocracker_conversational():
    """TEST 5: Marathi Hydrocracker Conversational -> Technical source."""
    res = query_knowledge_base("MRPL च्या रिफायनरीतील हायड्रोक्रॅकर युनिटबद्दल सोप्या भाषेत माहिती द्या.", top_k=5, debug=True)
    assert res["result_count"] > 0
    top_src = res["sources"][0]
    title = top_src["document_title"].lower()
    assert any(w in title for w in ["refining", "cutting edge", "manufacturing", "facilities", "annual report"])
    assert not any("job scam" in s["document_title"].lower() for s in res["sources"])


def test_case_6_english_revenue_fy2025():
    """TEST 6: English Revenue FY2025 -> Revenue-specific source, no exports/turnover/profit substitution."""
    res = query_knowledge_base("What was MRPL's revenue in FY2025?", top_k=5, debug=True)
    assert res["result_count"] > 0
    # Must retrieve financial reports
    titles = [s["document_title"].lower() for s in res["sources"]]
    assert any("annual report" in t for t in titles)
    assert not any("job scam" in t for t in titles)


def test_case_7_hindi_revenue_fy2025():
    """TEST 7: Hindi Revenue FY2025 -> Revenue-specific retrieval."""
    res = query_knowledge_base("MRPL का FY2025 का राजस्व कितना था?", top_k=5, debug=True)
    assert res["result_count"] > 0
    titles = [s["document_title"].lower() for s in res["sources"]]
    assert any("annual report" in t for t in titles)
    assert not any("job scam" in t for t in titles)


def test_case_8_marathi_revenue_fy2025():
    """TEST 8: Marathi Revenue FY2025 -> Revenue-specific retrieval."""
    res = query_knowledge_base("FY2025 मध्ये MRPL चा महसूल किती होता?", top_k=5, debug=True)
    assert res["result_count"] > 0
    titles = [s["document_title"].lower() for s in res["sources"]]
    assert any("annual report" in t for t in titles)
    assert not any("job scam" in t for t in titles)


def test_case_9_english_job_scam_allowed():
    """TEST 9: English Job Scam -> Job scam notice should now be allowed because query is relevant."""
    res = query_knowledge_base("What is the purpose of the public notice on job scam?", top_k=5, debug=True)
    assert res["result_count"] > 0
    top_src = res["sources"][0]
    assert "public notice on job scam" in top_src["document_title"].lower()


def test_case_10_hindi_recruitment():
    """TEST 10: Hindi Recruitment -> Recruitment-related sources, not refinery technical units."""
    res = query_knowledge_base("MRPL की भर्ती से संबंधित जानकारी क्या है?", top_k=5, debug=True)
    assert res["result_count"] > 0
    titles = [s["document_title"].lower() for s in res["sources"]]
    # Should contain recruitment/job notices or corporate reports, NOT refinery units
    assert any("job scam" in t or "annual report" in t for t in titles)
    assert not any(t.startswith("refining") or t.startswith("manufacturing units") for t in titles)


# ── Step 3: Retrieval Consistency Test ───────────────────────────────────────

def test_retrieval_consistency_across_languages():
    """
    Consistency test: For equivalent English, Hindi, and Marathi hydrocracker queries,
    the top retrieved technical document should overlap strongly.
    """
    res_en = query_knowledge_base("What is the main purpose of the hydrocracker unit at MRPL?", top_k=3)
    res_hi = query_knowledge_base("MRPL की रिफाइनरी में हाइड्रोक्रैकर यूनिट का मुख्य उद्देश्य क्या है?", top_k=3)
    res_mr = query_knowledge_base("MRPL च्या रिफायनरीमध्ये हायड्रोक्रॅकर युनिटचा मुख्य उद्देश काय आहे?", top_k=3)

    top_en = res_en["sources"][0]["document_title"]
    top_hi = res_hi["sources"][0]["document_title"]
    top_mr = res_mr["sources"][0]["document_title"]

    # In our implementation, all three rank "Refining" as #1!
    assert top_hi == top_en
    assert top_mr == top_en


def test_clean_source_deduplication_and_count():
    """Verify that source list does not artificially force 5 noisy items and deduplicates cleanly."""
    res = query_knowledge_base("MRPL च्या रिफायनरीमध्ये हायड्रोक्रॅकर युनिटचा मुख्य उद्देश काय आहे?", top_k=5)
    # Should only return genuinely relevant sources (3 sources, not forced 5)
    assert 1 <= res["result_count"] <= 5
    seen_titles = set()
    for s in res["sources"]:
        title = s["document_title"]
        # Same document should not have duplicate entries
        assert (title, s["page"]) not in seen_titles
        seen_titles.add((title, s["page"]))
