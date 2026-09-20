"""
Tests for document generation (DOCX, XLSX, PPTX).
"""

from pathlib import Path
from backend.services.docgen import generate_docx, generate_xlsx, generate_pptx


def test_generate_docx():
    title = "MRPL Test Safety Report"
    content = "Executive Summary\n\nAll refining units operated within standard parameters."
    res = generate_docx(title, content)

    assert "path" in res
    assert res["filename"].endswith(".docx")
    assert res["file_size_bytes"] > 0
    assert Path(res["path"]).exists()


def test_generate_xlsx():
    title = "MRPL Production Metrics"
    headers = ["Unit", "Feedstock (TMT)", "Capacity (%)"]
    data = [
        ["CDU-1", 450.5, 98.2],
        ["CDU-2", 480.0, 101.4],
        ["PFCCU", 210.3, 95.0],
    ]
    res = generate_xlsx(title, data, headers=headers)

    assert "path" in res
    assert res["filename"].endswith(".xlsx")
    assert res["file_size_bytes"] > 0
    assert Path(res["path"]).exists()


def test_generate_pptx():
    title = "MRPL Technical Review"
    slides = [
        {"title": "Introduction", "bullet_points": ["Air-gapped deployment", "Local Ollama models"]},
        {"title": "Performance", "bullet_points": ["Zero external egress", "ChromaDB RAG search"]},
    ]
    res = generate_pptx(title, slides)

    assert "path" in res
    assert res["filename"].endswith(".pptx")
    assert res["file_size_bytes"] > 0
    assert Path(res["path"]).exists()


def test_generate_pdf():
    from backend.services.docgen import generate_pdf, verify_pdf
    title = "DOCUMENT ANALYSIS REPORT"
    content = "1. Executive Summary\nAnalysis of TranscriptSd.pdf.\n\n2. Key Findings\nGross crude 4.56, net crude 4.7 MMT.\n\n3. Detailed Analysis\nDiscussion on operational parameters.\n\n4. Evidence / Page References\nPage 12.\n\n5. Conclusion\nSuccessful review."
    res = generate_pdf(title=title, content=content, document_name="TranscriptSd.pdf")

    assert "path" in res
    assert res["filename"] == "TranscriptSd_Analysis.pdf"
    assert res["file_size_bytes"] > 0
    p = Path(res["path"])
    assert p.exists()
    is_valid, msg = verify_pdf(p)
    assert is_valid, msg


def test_generate_docx_with_document_name():
    from backend.services.docgen import generate_docx, verify_docx
    title = "DOCUMENT ANALYSIS REPORT"
    content = "1. Executive Summary\nAnalysis of TranscriptSd.pdf.\n\n2. Key Findings\nGross crude 4.56, net crude 4.7 MMT.\n\n3. Detailed Analysis\nDiscussion on operational parameters.\n\n4. Evidence / Page References\nPage 12.\n\n5. Conclusion\nSuccessful review."
    res = generate_docx(title=title, content=content, document_name="TranscriptSd.pdf")

    assert "path" in res
    assert res["filename"] == "TranscriptSd_Analysis.docx"
    assert res["file_size_bytes"] > 0
    p = Path(res["path"])
    assert p.exists()
    is_valid, msg = verify_docx(p)
    assert is_valid, msg


SAMPLE_LLM_ANALYSIS = """## Document Analysis: TranscriptSd.pdf

**1. Executive Summary**

This document is a transcript of a discussion regarding operational and financial aspects of MRPL, specifically concerning the IBB project and broader financial considerations. The core takeaway is the reliance on net crude volume as the primary driver of revenue and profitability, alongside a cautious outlook on the IBB project's IRR due to its early stage and reliance on regulatory approvals and market conditions.

**2. Key Findings**

*   The gross crude volume reported was 4.56 Million Metric Tons, representing the net crude percentage. (Page 13)
*   The IBB project is a pilot project expected to take 3-4 years to reach commercialization. (Page 19)
*   The IRR for the IBB project is currently unfeasible to determine due to the project's stage and external factors. (Page 19)

**3. Detailed Analysis**

The transcript primarily focuses on clarifying MRPL's financial reporting methodology and the status of the IBB project. The discussion highlights that MRPL's revenue and bottom line are predominantly influenced by the volume of net crude processed, rather than the gross crude volume. This suggests a strategic emphasis on operational efficiency and refined product output.

Regarding the IBB project, the participants acknowledge its early stage, emphasizing the numerous decision gates and stringent quality norms involved. The timeline for commercialization is estimated at 3-4 years, contingent on obtaining licenses and market conditions. The current inability to assess the IRR reflects the uncertainty surrounding these factors.

**4. Evidence / Page References**

*   **Page 13:** "So, the thing is, Avin this side, the 4.56 that you mentioned was the gross crude, the 4.7 Million Metric Ton that we reported in our financials that was the net crude percentage."
*   **Page 19:** "So regarding IBB it's a pilot project and we are still quite a few years from its commercialization. So we need to go across many decision gates. First is the technical inspection of that. It has to meet stringent quality norms. And we are quite hopeful of that. After that to get the licenses we still see about 3-4 years down the line. And only after that we'll be in a position to take a realistic view on the IRR."

**5. Conclusion**

The transcript provides insights into MRPL's financial reporting practices and the development of the IBB project. The company's focus on net crude volume as a key performance indicator is established. Furthermore, the IBB project's timeline and the current inability to determine its IRR underscore the project's early stage and the influence of external factors on its future profitability. Recommended follow-up actions include monitoring the progress of the IBB project's regulatory approvals and continued assessment of market conditions to refine the IRR projection.

Sources:
1. TranscriptSd, p. 13
2. TranscriptSd, p. 19
"""


def test_parse_analysis_report():
    from backend.services.docgen import parse_analysis_report

    report = parse_analysis_report(SAMPLE_LLM_ANALYSIS, document_name="TranscriptSd.pdf")

    assert report.title == "DOCUMENT ANALYSIS REPORT"
    assert report.subtitle == "Document Analysis: TranscriptSd.pdf"
    assert report.document_name == "TranscriptSd.pdf"

    # Exactly 5 canonical sections
    assert len(report.sections) == 5
    section_titles = [s.title for s in report.sections]
    assert section_titles == [
        "1. Executive Summary",
        "2. Key Findings",
        "3. Detailed Analysis",
        "4. Evidence / Page References",
        "5. Conclusion",
    ]

    # Sources parsed
    assert len(report.sources) == 2
    assert "TranscriptSd, p. 13" in report.sources[0]
    assert "TranscriptSd, p. 19" in report.sources[1]


def test_transcript_sd_clean_generation():
    import pymupdf
    from docx import Document
    from backend.services.docgen import generate_pdf, generate_docx

    # Generate PDF
    pdf_res = generate_pdf(
        title="DOCUMENT ANALYSIS REPORT",
        content=SAMPLE_LLM_ANALYSIS,
        document_name="TranscriptSd.pdf",
        filename="TranscriptSd_Analysis.pdf",
    )
    pdf_path = Path(pdf_res["path"])
    assert pdf_path.exists()
    assert pdf_res["file_size_bytes"] > 0

    # Inspect PDF contents
    with pymupdf.open(str(pdf_path)) as pdf_doc:
        assert pdf_doc.page_count >= 1
        pdf_text = "".join(page.get_text() for page in pdf_doc)

    # 1. Check sections appear EXACTLY ONCE
    for sec in ["Executive Summary", "Key Findings", "Detailed Analysis", "Evidence / Page References", "Conclusion"]:
        count = pdf_text.count(sec)
        assert count == 1, f"Expected 1 occurrence of '{sec}' in PDF, found {count}"

    # 2. Check ZERO raw Markdown in PDF
    for md in ["##", "**", "---"]:
        assert md not in pdf_text, f"Found raw Markdown '{md}' in rendered PDF"

    # 3. Check key facts preserved in PDF
    pdf_norm = " ".join(pdf_text.split())
    assert "4.56 Million Metric Tons" in pdf_norm
    assert "4.7 Million Metric Ton" in pdf_norm
    assert "IBB project is a pilot project" in pdf_norm
    assert "3-4 years" in pdf_norm
    assert "IRR" in pdf_norm
    assert "Page 13" in pdf_norm
    assert "Page 19" in pdf_norm

    # Generate DOCX
    docx_res = generate_docx(
        title="DOCUMENT ANALYSIS REPORT",
        content=SAMPLE_LLM_ANALYSIS,
        document_name="TranscriptSd.pdf",
        filename="TranscriptSd_Analysis.docx",
    )
    docx_path = Path(docx_res["path"])
    assert docx_path.exists()
    assert docx_res["file_size_bytes"] > 0

    # Inspect DOCX contents
    docx_doc = Document(str(docx_path))
    docx_text = "\n".join(p.text for p in docx_doc.paragraphs)

    # 1. Check sections appear EXACTLY ONCE
    for sec in ["Executive Summary", "Key Findings", "Detailed Analysis", "Evidence / Page References", "Conclusion"]:
        count = docx_text.count(sec)
        assert count == 1, f"Expected 1 occurrence of '{sec}' in DOCX, found {count}"

    # 2. Check ZERO raw Markdown in DOCX
    for md in ["##", "**", "---"]:
        assert md not in docx_text, f"Found raw Markdown '{md}' in rendered DOCX"

    # 3. Check key facts preserved in DOCX
    docx_norm = " ".join(docx_text.split())
    assert "4.56 Million Metric Tons" in docx_norm
    assert "4.7 Million Metric Ton" in docx_norm
    assert "IBB project is a pilot project" in docx_norm
    assert "3-4 years" in docx_norm
    assert "IRR" in docx_norm
    assert "Page 13" in docx_norm
    assert "Page 19" in docx_norm


