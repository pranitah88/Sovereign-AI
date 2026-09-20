"""
Regression Tests for Response Language and Technical Entity Fidelity.

Requirements:
A. Query "what are the main units in mrpl" produces an English response with no Marathi script,
   preserving official technical entities (Platforming, Delayed Coker Unit, Bitumen, PFCCU, etc.).
B. Marathi query ("MRPL मधील मुख्य युनिट्स कोणते आहेत?") preserves technical names in official terminology.
C. Hindi query ("MRPL की मुख्य यूनिट्स कौन सी हैं?") preserves technical names in official terminology.
D. Technical acronym preservation: PFCCU remains PFCCU, not पीएफसीयू.
E. Technical name preservation: Delayed Coker Unit remains identifiable as Delayed Coker Unit.
F. Current-query language isolation: Previous conversation in Marathi + current query in English
   produces an English answer.
"""

import pytest
import re
from backend.services.entity_preservation import (
    detect_query_language_deterministic,
    extract_technical_entities,
    repair_transliterated_technical_terms,
    validate_and_preserve_entities,
)
from backend.agent.prompts import REASONING_PROMPT


def test_query_a_english_language_and_technical_entity_fidelity():
    """Test A: Query 'what are the main units in mrpl' -> English, no Marathi, preserved entities."""
    query = "what are the main units in mrpl"
    lang = detect_query_language_deterministic(query)
    assert lang == "en", f"Expected 'en', got '{lang}'"

    # Simulated response with both valid entities and a check that transliterations are caught
    evidence = (
        "The main Secondary Processing Units in MRPL are: Hydrocracker Unit, "
        "Continuous Catalytic Regeneration Platforming Unit, Isomerisation Unit, "
        "Petro Fluidised Catalytic Cracking Unit (PFCCU), Delayed Coker Unit, "
        "Visbreaker Unit, Bitumen Unit, Treating Unit."
    )
    detected_entities = extract_technical_entities(evidence)
    assert "Delayed Coker Unit" in detected_entities
    assert "Platforming Unit" in detected_entities or "Platforming" in detected_entities
    assert "Bitumen Unit" in detected_entities or "Bitumen" in detected_entities
    assert "PFCCU" in detected_entities

    # If the response had Marathi or transliterations, validator repairs it
    corrupt_response = (
        "MRPL मध्ये मुख्य युनिट्स खालीलप्रमाणे आहेत: क्रूड युनिट, हायड्रोक्रॅकर युनिट, "
        "प्लॅटफासिंग युनिट, डिलेस्ड कोकर युनिट, बिटमॅन युनिट, पीएफसीयू. (Source: Refining, p. 1)"
    )
    result = validate_and_preserve_entities(corrupt_response, query, evidence)
    assert result.query_language == "en"
    assert result.response_language == "en"
    # Ensure no Devanagari remains in final response
    assert not re.search(r"[\u0900-\u097F]", result.corrected_response), (
        f"Devanagari found in English response: {result.corrected_response}"
    )
    # Ensure technical terms are restored
    assert "Delayed Coker Unit" in result.corrected_response
    assert "Platforming" in result.corrected_response
    assert "Bitumen" in result.corrected_response
    assert "PFCCU" in result.corrected_response


def test_query_b_marathi_query_preserves_technical_names():
    """Test B: Query in Marathi preserves technical names in English/source terminology."""
    query = "MRPL मधील मुख्य युनिट्स कोणते आहेत?"
    lang = detect_query_language_deterministic(query)
    assert lang == "mr", f"Expected 'mr', got '{lang}'"

    evidence = "Delayed Coker Unit, Platforming Unit, Bitumen Unit, PFCCU"
    raw_marathi_response = (
        "MRPL मधील मुख्य युनिट्स खालीलप्रमाणे आहेत: प्लॅटफासिंग युनिट, डिलेस्ड कोकर युनिट, बिटमॅन युनिट आणि पीएफसीयू."
    )
    result = validate_and_preserve_entities(raw_marathi_response, query, evidence)
    assert result.query_language == "mr"
    assert result.response_language == "mr"
    # Marathi grammar is preserved, but technical entities are restored to official English names
    assert "Platforming" in result.corrected_response
    assert "Delayed Coker" in result.corrected_response
    assert "Bitumen" in result.corrected_response
    assert "PFCCU" in result.corrected_response
    assert "प्लॅटफासिंग" not in result.corrected_response
    assert "डिलेस्ड कोकर" not in result.corrected_response


def test_query_c_hindi_query_preserves_technical_names():
    """Test C: Query in Hindi preserves technical names in English/source terminology."""
    query = "MRPL की मुख्य यूनिट्स कौन सी हैं?"
    lang = detect_query_language_deterministic(query)
    assert lang == "hi", f"Expected 'hi', got '{lang}'"

    evidence = "Delayed Coker Unit, Platforming Unit, Bitumen Unit, PFCCU"
    raw_hindi_response = (
        "MRPL की मुख्य यूनिट्स निम्नलिखित हैं: प्लॅटफासिंग यूनिट, डिलेस्ड कोकर यूनिट, बिटमॅन यूनिट और पीएफसीयू।"
    )
    result = validate_and_preserve_entities(raw_hindi_response, query, evidence)
    assert result.query_language == "hi"
    assert result.response_language == "hi"
    assert "Platforming" in result.corrected_response
    assert "Delayed Coker" in result.corrected_response
    assert "Bitumen" in result.corrected_response
    assert "PFCCU" in result.corrected_response


def test_query_d_acronym_pfccu_preservation():
    """Test D: Technical acronym PFCCU must remain PFCCU, not पीएफसीयू."""
    text_with_corrupt_acronym = "MRPL operates a modern पीएफसीयू for light olefins production."
    repaired, repaired_terms = repair_transliterated_technical_terms(text_with_corrupt_acronym)
    assert "PFCCU" in repaired
    assert "पीएफसीयू" not in repaired


def test_query_e_technical_name_delayed_coker_unit_preservation():
    """Test E: Technical name Delayed Coker Unit must remain exactly identifiable as Delayed Coker Unit."""
    text_with_corrupt_unit = "Upgrading residue is handled by डिलेस्ड कोकर युनिट."
    repaired, repaired_terms = repair_transliterated_technical_terms(text_with_corrupt_unit)
    assert "Delayed Coker Unit" in repaired
    assert "डिलेस्ड कोकर" not in repaired


def test_query_f_current_query_language_isolation():
    """Test F: Current-query language isolation: Previous conversation in Marathi + current in English -> English."""
    current_query = "what are the main units in mrpl"
    # Even if chat history is full of Marathi, current query language MUST be English
    detected = detect_query_language_deterministic(current_query)
    assert detected == "en", "Current query language must be detected strictly from current query"

    # Verify prompt formatting generates strict English instruction regardless of prior Marathi chat
    marathi_chat_history = "user: MRPL मधील मुख्य युनिट्स सांगा\nassistant: MRPL मध्ये अनेक युनिट्स आहेत."
    formatted_prompt = REASONING_PROMPT.format(
        current_query=current_query,
        conversation_history=marathi_chat_history,
        retrieved_context="Refining Units: Hydrocracker, Delayed Coker Unit, Platforming, Bitumen, PFCCU",
    )
    assert "MANDATORY LANGUAGE: English (en)" in formatted_prompt
    assert "Do NOT output any Marathi, Hindi, or Devanagari script" in formatted_prompt


def test_trace_fields_structure():
    """Verify trace fields conform to the specification in Section 9."""
    query = "what are the main units in mrpl"
    evidence = "Refining document mentioning Delayed Coker Unit and Platforming."
    clean_resp = "The main units include Delayed Coker Unit and Platforming."
    result = validate_and_preserve_entities(clean_resp, query, evidence)
    trace_data = result.to_dict()

    assert trace_data["QUERY_LANGUAGE"] == "en"
    assert trace_data["RESPONSE_LANGUAGE"] == "en"
    assert trace_data["LANGUAGE_DECISION_SOURCE"] == "current_user_query"
    assert "Delayed Coker Unit" in trace_data["TECHNICAL_ENTITIES_DETECTED"]
    assert "Platforming" in trace_data["TECHNICAL_ENTITIES_DETECTED"]
    assert "Delayed Coker Unit" in trace_data["TECHNICAL_ENTITIES_PRESERVED"]
    assert "Platforming" in trace_data["TECHNICAL_ENTITIES_PRESERVED"]
    assert trace_data["TECHNICAL_ENTITY_VALIDATION"] == "PASSED"
