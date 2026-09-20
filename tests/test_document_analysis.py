"""
Regression tests for Document Analysis Pipeline in MRPL Sovereign AI Workbench.
Verifies complete document coverage, page reference validation, audience guardrails,
and strict source grounding on "Online Vigilance Complaint Portal Help Manual.pdf".
"""

import pytest
from pathlib import Path
import pymupdf
from docx import Document

from backend.services.doc_analysis import (
    DocumentModel,
    DocumentPage,
    resolve_document_file,
    extract_full_document,
    build_document_analysis_context,
    validate_and_sanitize_analysis,
    normalize_doc_name,
)
from backend.services.task_router import route_task
from backend.services.docgen import (
    generate_pdf,
    generate_docx,
    parse_analysis_report,
    verify_pdf,
    verify_docx,
)

VIGILANCE_DOC_NAME = "Online Vigilance Complaint Portal Help Manual.pdf"


def test_01_document_resolution():
    """Verify resolver accurately locates the test document on disk."""
    path, meta = resolve_document_file(VIGILANCE_DOC_NAME)
    assert path is not None, f"Could not resolve {VIGILANCE_DOC_NAME}"
    assert path.exists(), f"Path does not exist: {path}"
    assert path.suffix.lower() == ".pdf"


def test_02_full_document_coverage_and_page_count():
    """
    Verify 3-page document is recognized correctly with 100% complete coverage:
    - Exactly 3 pages extracted
    - Every page has text
    - Page 2 contains 20 MB attachment limit
    - Page 3 contains all critical notes
    """
    model = extract_full_document(VIGILANCE_DOC_NAME)
    assert model.extraction_status == "success", f"Extraction failed: {model.error_message}"
    assert model.total_pages == 3, f"Expected 3 pages, got {model.total_pages}"
    assert len(model.pages) == 3

    # Page 1: Login with OTP
    p1 = model.pages[0]
    assert p1.page_num == 1
    assert "login with otp" in p1.text.lower()
    assert "email" in p1.text.lower()

    # Page 2: Attachments & 20 MB limit
    p2 = model.pages[1]
    assert p2.page_num == 2
    assert "20 mb" in p2.text.lower()
    assert ".jpg" in p2.text.lower()
    assert ".pdf" in p2.text.lower()

    # Page 3: Tracking & Important Notes (CVC, CVO-MRPL, 15 days, PIDPI, confidentiality)
    p3 = model.pages[2]
    assert p3.page_num == 3
    p3_lower = p3.text.lower()
    p3_norm = " ".join(p3_lower.split())
    assert "tracking number" in p3_norm
    assert "complaint tracking" in p3_norm
    assert "cvc" in p3_norm
    assert "cvo" in p3_norm
    assert "15 day" in p3_norm or "15-day" in p3_norm
    assert "secret" in p3_norm
    assert "not saved" in p3_norm
    assert "pidpi" in p3_norm


def test_03_context_builder_bounds():
    """Verify built context declares total pages and labels pages 1 to 3."""
    model = extract_full_document(VIGILANCE_DOC_NAME)
    ctx = build_document_analysis_context(model)

    assert "TOTAL PAGES: 3" in ctx
    assert "(Valid page references are strictly Pages 1 to 3)" in ctx
    assert "--- [PAGE 1 OF 3] ---" in ctx
    assert "--- [PAGE 2 OF 3] ---" in ctx
    assert "--- [PAGE 3 OF 3] ---" in ctx


def test_04_page_reference_validation_and_remapping():
    """
    Simulate the exact problem found where LLM generated Page 11, Page 12, Page 13
    due to numbered steps 11, 12, 13 on page 3, and asserted 'for MRPL employees'.
    Verify sanitizer remaps citations to Page 3 and removes audience assumptions.
    """
    model = extract_full_document(VIGILANCE_DOC_NAME)

    raw_hallucinated_analysis = (
        "## Document Analysis: Online Vigilance Complaint Portal Help Manual.pdf\n\n"
        "A. Executive Summary\n"
        "This document provides a guide for MRPL employees on how to use the Online Vigilance Complaint Portal.\n\n"
        "B. Key Findings\n"
        "- Combined attachment limit is 20 MB.\n"
        "- Tracking number is sent to complainant.\n\n"
        "C. Detailed Analysis\n"
        "Users enter personal details on Page 2. A tracking number is provided upon submission (Page 11). "
        "Users navigate to Complaint Tracking (Page 12) and input their complaint number (Page 13).\n\n"
        "D. Important Notes / Exceptions\n"
        "- 15-day confirmation is required.\n\n"
        "E. Evidence / Page References\n"
        "Page 2: Personal details and 20 MB attachment limit.\n"
        "Page 11: Tracking number delivered.\n"
        "Page 12: Complaint Tracking tab.\n"
        "Page 13: View complaint status.\n\n"
        "F. Analysis / Interpretation\n"
        "The system enforces strict size and confirmation controls.\n\n"
        "G. Recommendations\n"
        "Regularly review complaint logs."
    )

    sanitized = validate_and_sanitize_analysis(raw_hallucinated_analysis, model)
    s_lower = sanitized.lower()

    # Verify ZERO hallucinated page numbers
    assert "page 11" not in s_lower, "Page 11 was not eliminated!"
    assert "page 12" not in s_lower, "Page 12 was not eliminated!"
    assert "page 13" not in s_lower, "Page 13 was not eliminated!"

    # Verify remapped to Page 3
    assert "page 3" in s_lower

    # Verify audience guardrail: 'MRPL employees' replaced
    assert "for mrpl employees" not in s_lower
    assert "mrpl employees" not in s_lower

    # Verify mandatory Page 3 notes enriched if missing
    assert "cvo" in s_lower
    assert "secret" in s_lower or "confidential" in s_lower
    assert "pidpi" in s_lower


def test_05_docgen_pdf_and_docx_rendering():
    """Verify clean PDF and DOCX generation with all 7 structured sections."""
    model = extract_full_document(VIGILANCE_DOC_NAME)

    analysis_content = (
        "A. Executive Summary\n"
        "This document outlines the user procedures for lodging and tracking complaints on the Online Vigilance Complaint Portal.\n\n"
        "B. Key Findings\n"
        "- Login with OTP using Email ID or Phone number.\n"
        "- Maximum attachment size across all files is 20 MB (.jpg and .pdf supported).\n"
        "- Unique Tracking number issued upon submission.\n\n"
        "C. Detailed Analysis\n"
        "Complainants register with OTP, provide contact details, write complaint specifics, attach evidence, and submit.\n\n"
        "D. Important Notes / Exceptions\n"
        "- 15-Day Confirmation Requirement: Complainant must confirm submission within 15 days as per CVC guidelines (Page 3).\n"
        "- System Data Protection: Complainant details are NOT saved in the portal system; details submit directly to CVO–MRPL (Page 3).\n"
        "- Confidentiality: Complainant identity is kept strictly secret (Page 3).\n"
        "- PIDPI Mechanism: Direct complaints to CVC for identity protection must be submitted by Post under PIDPI (Page 3).\n\n"
        "E. Evidence / Page References\n"
        "Page 1: Login with OTP and contact entry.\n"
        "Page 2: Attachment details and 20 MB size limit.\n"
        "Page 3: Tracking number, 15-day confirmation, CVO–MRPL direct submission, PIDPI.\n\n"
        "F. Analysis / Interpretation\n"
        "The architecture prioritizes whistleblower protection by preventing database storage of personal details.\n\n"
        "G. Recommendations\n"
        "Portal administrators should ensure mail server reliability to prevent OTP delivery delays."
    )

    # 1. Parse report structure
    report = parse_analysis_report(analysis_content, document_name=VIGILANCE_DOC_NAME)
    sec_titles = [s.title for s in report.sections]
    assert "A. Executive Summary" in sec_titles
    assert "B. Key Findings" in sec_titles
    assert "C. Detailed Analysis" in sec_titles
    assert "D. Important Notes / Exceptions" in sec_titles
    assert "E. Evidence / Page References" in sec_titles
    assert "F. Analysis / Interpretation" in sec_titles
    assert "G. Recommendations" in sec_titles

    # 2. Render PDF
    pdf_res = generate_pdf(
        title="DOCUMENT ANALYSIS REPORT",
        content=analysis_content,
        document_name=VIGILANCE_DOC_NAME,
        filename="Vigilance_Analysis_Test.pdf",
    )
    assert pdf_res["file_size_bytes"] > 0
    pdf_path = Path(pdf_res["path"])
    is_valid, msg = verify_pdf(pdf_path)
    assert is_valid, msg

    with pymupdf.open(str(pdf_path)) as p_doc:
        full_pdf = "".join(page.get_text() for page in p_doc)
        assert "Executive Summary" in full_pdf
        assert "Key Findings" in full_pdf
        assert "Detailed Analysis" in full_pdf
        assert "Important Notes / Exceptions" in full_pdf
        assert "Evidence / Page References" in full_pdf
        assert "Analysis / Interpretation" in full_pdf
        assert "Recommendations" in full_pdf
        assert "20 MB" in full_pdf
        assert "CVO" in full_pdf
        assert "PIDPI" in full_pdf
        assert "Page 11" not in full_pdf
        assert "Page 12" not in full_pdf

    # 3. Render DOCX
    docx_res = generate_docx(
        title="DOCUMENT ANALYSIS REPORT",
        content=analysis_content,
        document_name=VIGILANCE_DOC_NAME,
        filename="Vigilance_Analysis_Test.docx",
    )
    assert docx_res["file_size_bytes"] > 0
    docx_path = Path(docx_res["path"])
    is_valid_docx, msg_docx = verify_docx(docx_path)
    assert is_valid_docx, msg_docx


def test_06_controlled_limitation_on_missing_file():
    """Verify controlled refusal without hallucination when document is unavailable."""
    fake_model = extract_full_document("NonExistent_Report_File_XYZ.pdf")
    assert fake_model.total_pages == 0
    assert fake_model.extraction_status == "failed"

    sanitized = validate_and_sanitize_analysis("Fake analysis of non existent document", fake_model)
    assert "could not be reliably extracted" in sanitized
    assert "Controlled Limitation" in sanitized


def test_07_task_router_quoted_and_spaces():
    """Verify task router correctly identifies DOCUMENT_ANALYSIS for multi-word filenames."""
    queries = [
        f'analyze "{VIGILANCE_DOC_NAME}"',
        f"analyze {VIGILANCE_DOC_NAME}",
        f"summarize {VIGILANCE_DOC_NAME}",
        f"generate pdf of {VIGILANCE_DOC_NAME}",
    ]
    for q in queries:
        decision = route_task(q)
        assert decision["task_type"] == "DOCUMENT_ANALYSIS", f"Failed to route '{q}'"
        assert decision["target_document"] is not None
        assert "Vigilance" in decision["target_document"]
