"""
Document generation service — creates PDF, Word, Excel, and PowerPoint files.
Strictly verifies file existence, size, and readability before returning success.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import io
import logging
from pathlib import Path
import re
from typing import List, Tuple, Optional, Any
import uuid

from PIL import Image as PILImage
from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from openpyxl import Workbook
from pptx import Presentation

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Output directory for generated files (strictly within outputs directory)
_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def _ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def _safe_filename(name: str, ext: str = "") -> str:
    """Convert a name or title to a safe filename, strictly stripping directory traversal characters."""
    base = Path(name).name
    # Strip known extensions if present to avoid duplicate extensions
    for known_ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".csv", ".txt"):
        if base.lower().endswith(known_ext):
            base = base[:-len(known_ext)]
            break
    if ext:
        ext_clean = ext if ext.startswith(".") else f".{ext}"
        if base.lower().endswith(ext_clean.lower()):
            base = base[:-len(ext_clean)]
    cleaned = base.replace("/", "_").replace("\\", "_").replace("..", "_")
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_", ".") else "_" for c in cleaned)
    safe = safe.strip().replace(" ", "_")[:100] or "document"
    if ext:
        ext_clean = ext if ext.startswith(".") else f".{ext}"
        safe = f"{safe}{ext_clean}"
    return safe



def _resolve_output_path(filename: str) -> Path:
    """Resolve an output filename strictly inside the designated output directory."""
    out_dir = _ensure_output_dir().resolve()
    target = (out_dir / filename).resolve()
    if not str(target).startswith(str(out_dir)):
        raise ValueError(f"Directory traversal detected for filename: {filename}")
    return target


def verify_pdf(path: Path) -> tuple[bool, str]:
    """
    Verify that a generated PDF file exists, is non-empty, and can be read by PyMuPDF.
    Returns (is_valid, error_message).
    """
    if not path.exists():
        return False, f"PDF file does not exist on disk: {path}"
    if not path.is_file():
        return False, f"PDF path is not a regular file: {path}"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"Failed to stat PDF file: {exc}"
    if size <= 0:
        return False, f"PDF file is empty (size 0 bytes): {path}"

    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            if doc.page_count < 1:
                return False, f"PDF contains 0 pages: {path}"
            # Verify the first page can render text or inspect catalog
            _ = doc[0].get_text()
    except Exception as exc:
        return False, f"PDF readability verification failed: {exc}"

    return True, "Valid PDF"


def verify_docx(path: Path) -> tuple[bool, str]:
    """
    Verify that a generated DOCX file exists, is non-empty, and can be read by python-docx.
    Returns (is_valid, error_message).
    """
    if not path.exists():
        return False, f"DOCX file does not exist on disk: {path}"
    if not path.is_file():
        return False, f"DOCX path is not a regular file: {path}"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"Failed to stat DOCX file: {exc}"
    if size <= 0:
        return False, f"DOCX file is empty (size 0 bytes): {path}"

    try:
        doc = DocxDocument(str(path))
        if len(doc.paragraphs) == 0 and len(doc.tables) == 0:
            return False, f"DOCX contains no content paragraphs: {path}"
    except Exception as exc:
        return False, f"DOCX readability verification failed: {exc}"

    return True, "Valid DOCX"


@dataclass
class ReportElement:
    kind: str  # "paragraph", "bullet", "numbered", "subheading"
    raw_text: str
    clean_text: str
    html_text: str
    docx_runs: list[tuple[str, bool, bool]]  # (text, is_bold, is_italic)
    number: int | None = None


@dataclass
class ReportSection:
    title: str
    elements: list[ReportElement] = field(default_factory=list)


@dataclass
class ParsedReport:
    title: str
    subtitle: str | None
    document_name: str
    organization: str
    environment: str
    timestamp: str
    notice: str
    sections: list[ReportSection]
    sources: list[str] = field(default_factory=list)
    analysis_id: str = ""
    image_hash: str = ""
    analysis_source: str = ""
    verification_status: str = ""
    request_query: str = ""
    image_bytes: bytes | None = None
    image_path: str | None = None


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' in footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()  # type: ignore[attr-defined]

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        # Margin: left 54, right 612-54=558
        self.line(54, 36, 612 - 54, 36)
        self.drawString(54, 24, "MRPL Sovereign AI Workbench | Confidential & Proprietary")
        page_num = getattr(self, "_pageNumber", 1)
        page_str = f"Page {page_num} of {page_count}"
        self.drawRightString(612 - 54, 24, page_str)
        self.restoreState()


def markdown_to_reportlab_html(text: str) -> str:
    """Convert markdown text to safe ReportLab HTML formatted string."""
    t = re.sub(r"^#+\s*", "", text)
    # Escape XML entities first
    s = html.escape(t)
    # Bold + Italic: ***text***
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", s)
    # Bold: **text**
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # Italic: *text*
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    # Inline code
    s = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)
    return s.strip()


def parse_markdown_runs(text: str) -> list[tuple[str, bool, bool]]:
    """Parse a markdown text line into a list of (text_chunk, is_bold, is_italic)."""
    t = re.sub(r"^#+\s*", "", text)
    pattern = re.compile(r"(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*)")
    parts = pattern.split(t)
    runs = []
    for part in parts:
        if not part:
            continue
        if part.startswith("***") and part.endswith("***") and len(part) >= 6:
            runs.append((part[3:-3], True, True))
        elif part.startswith("**") and part.endswith("**") and len(part) >= 4:
            runs.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
            runs.append((part[1:-1], False, True))
        else:
            runs.append((part, False, False))
    return runs


def clean_plain_text(text: str) -> str:
    """Strip all markdown formatting markers from text for pure string representations."""
    t = re.sub(r"^#+\s*", "", text)
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"^\s*[-*_]{3,}\s*$", "", t)
    return t.strip()


SECTION_CANONICAL_PATTERNS = [
    ("1. Executive Summary", r"^(?:#+\s*|\*\*)?\s*1[\.\)]\s*Executive Summary(?:\*\*)?\s*$"),
    ("A. Executive Summary", r"^(?:#+\s*|\*\*)?\s*A[\.\)]\s*Executive Summary(?:\*\*)?\s*$"),
    ("1. Executive Summary", r"^(?:#+\s*|\*\*)?\s*Executive Summary(?:\*\*)?\s*$"),

    ("2. Key Findings", r"^(?:#+\s*|\*\*)?\s*2[\.\)]\s*Key Findings(?:\*\*)?\s*$"),
    ("B. Key Findings", r"^(?:#+\s*|\*\*)?\s*B[\.\)]\s*Key Findings(?:\*\*)?\s*$"),
    ("2. Key Findings", r"^(?:#+\s*|\*\*)?\s*Key Findings(?:\*\*)?\s*$"),

    ("3. Detailed Analysis", r"^(?:#+\s*|\*\*)?\s*3[\.\)]\s*Detailed Analysis(?:\*\*)?\s*$"),
    ("C. Detailed Analysis", r"^(?:#+\s*|\*\*)?\s*C[\.\)]\s*Detailed Analysis(?:\*\*)?\s*$"),
    ("3. Detailed Analysis", r"^(?:#+\s*|\*\*)?\s*Detailed Analysis(?:\*\*)?\s*$"),

    ("D. Important Notes / Exceptions", r"^(?:#+\s*|\*\*)?\s*D[\.\)]\s*(?:Important\s+Notes\s*(?:/|and|&)?\s*Exceptions|Important\s+Notes|Notes\s*(?:/|and|&)?\s*Exceptions|Exceptions)(?:\*\*)?\s*$"),
    ("Important Notes / Exceptions", r"^(?:#+\s*|\*\*)?\s*(?:4[\.\)]\s*)?(?:Important\s+Notes\s*(?:/|and|&)?\s*Exceptions|Important\s+Notes|Notes\s*(?:/|and|&)?\s*Exceptions|Exceptions)(?:\*\*)?\s*$"),

    ("4. Evidence / Page References", r"^(?:#+\s*|\*\*)?\s*4[\.\)]\s*(?:Evidence\s*(?:/|and|&)\s*Page References|Page References|Evidence)(?:\*\*)?\s*$"),
    ("E. Evidence / Page References", r"^(?:#+\s*|\*\*)?\s*E[\.\)]\s*(?:Evidence\s*(?:/|and|&)\s*Page References|Page References|Evidence)(?:\*\*)?\s*$"),
    ("5. Evidence / Page References", r"^(?:#+\s*|\*\*)?\s*5[\.\)]\s*(?:Evidence\s*(?:/|and|&)\s*Page References|Page References|Evidence)(?:\*\*)?\s*$"),
    ("4. Evidence / Page References", r"^(?:#+\s*|\*\*)?\s*(?:Evidence\s*(?:/|and|&)\s*Page References|Page References|Evidence)(?:\*\*)?\s*$"),

    ("F. Analysis / Interpretation", r"^(?:#+\s*|\*\*)?\s*F[\.\)]\s*(?:Analysis\s*(?:/|and|&)\s*Interpretation|Analysis\s*/\s*Interpretation|Interpretation)(?:\*\*)?\s*$"),
    ("Analysis / Interpretation", r"^(?:#+\s*|\*\*)?\s*(?:6[\.\)]\s*)?(?:Analysis\s*(?:/|and|&)\s*Interpretation|Analysis\s*/\s*Interpretation|Interpretation)(?:\*\*)?\s*$"),

    ("G. Recommendations", r"^(?:#+\s*|\*\*)?\s*G[\.\)]\s*Recommendations(?:\*\*)?\s*$"),
    ("Recommendations", r"^(?:#+\s*|\*\*)?\s*(?:7[\.\)]\s*)?Recommendations(?:\*\*)?\s*$"),

    ("5. Conclusion", r"^(?:#+\s*|\*\*)?\s*(?:5[\.\)]\s*)?Conclusion(?:\*\*)?\s*$"),
    ("Sources", r"^(?:#+\s*|\*\*)?\s*Sources:?(?:\*\*)?\s*$"),
]


def parse_analysis_report(
    content: str,
    document_name: str,
    title: str = "DOCUMENT ANALYSIS REPORT",
    timestamp: str = "",
) -> ParsedReport:
    """
    Parse an existing LLM analysis into a normalized report representation.
    Extracts subtitle, strips markdown syntax, normalizes section titles,
    and guarantees each section appears exactly once without duplicated wrapping.
    """
    lines = content.splitlines()
    subtitle = None
    sources = []

    sections_dict: dict[str, list[ReportElement]] = {}
    current_sec_title = None
    current_lines: list[str] = []

    def flush_lines_to_section(sec_title: str, lines_block: list[str]):
        if not sec_title or not lines_block:
            return
        if sec_title not in sections_dict:
            sections_dict[sec_title] = []

        i = 0
        while i < len(lines_block):
            line = lines_block[i].rstrip()
            trimmed = line.strip()
            if not trimmed:
                i += 1
                continue

            # Skip horizontal rule lines
            if re.match(r"^\s*[-*_]{3,}\s*$", trimmed):
                i += 1
                continue

            # Bullet points (*, -, •, +)
            bullet_match = re.match(r"^(\*|-|•|\+)\s+(.*)$", trimmed)
            if bullet_match:
                bullet_body = bullet_match.group(2).strip()
                i += 1
                while i < len(lines_block):
                    next_line = lines_block[i].rstrip()
                    next_trim = next_line.strip()
                    if not next_trim:
                        break
                    if re.match(r"^(\*|-|•|\+|\d+\.)\s+", next_trim) or any(
                        re.match(pat, next_trim, re.IGNORECASE) for _, pat in SECTION_CANONICAL_PATTERNS
                    ):
                        break
                    bullet_body += " " + next_trim
                    i += 1

                elem = ReportElement(
                    kind="bullet",
                    raw_text=bullet_body,
                    clean_text=clean_plain_text(bullet_body),
                    html_text=markdown_to_reportlab_html(bullet_body),
                    docx_runs=parse_markdown_runs(bullet_body),
                )
                sections_dict[sec_title].append(elem)
                continue

            # Numbered list (e.g. 1. , 2. )
            num_match = re.match(r"^(\d+)[\.\)]\s+(.*)$", trimmed)
            if num_match:
                num = int(num_match.group(1))
                num_body = num_match.group(2).strip()
                i += 1
                while i < len(lines_block):
                    next_line = lines_block[i].rstrip()
                    next_trim = next_line.strip()
                    if not next_trim:
                        break
                    if re.match(r"^(\*|-|•|\+|\d+\.)\s+", next_trim) or any(
                        re.match(pat, next_trim, re.IGNORECASE) for _, pat in SECTION_CANONICAL_PATTERNS
                    ):
                        break
                    num_body += " " + next_trim
                    i += 1

                elem = ReportElement(
                    kind="numbered",
                    raw_text=num_body,
                    clean_text=clean_plain_text(num_body),
                    html_text=markdown_to_reportlab_html(num_body),
                    docx_runs=parse_markdown_runs(num_body),
                    number=num,
                )
                sections_dict[sec_title].append(elem)
                continue

            # Subheadings inside a section (e.g. ### Subheading)
            if trimmed.startswith("### ") or trimmed.startswith("## "):
                sub_text = clean_plain_text(trimmed)
                elem = ReportElement(
                    kind="subheading",
                    raw_text=trimmed,
                    clean_text=sub_text,
                    html_text=markdown_to_reportlab_html(trimmed),
                    docx_runs=parse_markdown_runs(trimmed),
                )
                sections_dict[sec_title].append(elem)
                i += 1
                continue

            # Normal paragraphs
            para_lines = [trimmed]
            i += 1
            while i < len(lines_block):
                next_line = lines_block[i].rstrip()
                next_trim = next_line.strip()
                if not next_trim:
                    break
                if (
                    re.match(r"^(\*|-|•|\+|\d+\.)\s+", next_trim)
                    or any(re.match(pat, next_trim, re.IGNORECASE) for _, pat in SECTION_CANONICAL_PATTERNS)
                    or next_trim.startswith("##")
                ):
                    break
                para_lines.append(next_trim)
                i += 1

            para_text = " ".join(para_lines)
            elem = ReportElement(
                kind="paragraph",
                raw_text=para_text,
                clean_text=clean_plain_text(para_text),
                html_text=markdown_to_reportlab_html(para_text),
                docx_runs=parse_markdown_runs(para_text),
            )
            sections_dict[sec_title].append(elem)

    for line in lines:
        trimmed = line.strip()

        # Detect document subtitle at top
        if not subtitle and not current_sec_title:
            sub_match = re.match(
                r"^(?:#+\s*|\*\*)?(?:Document\s+Analysis|Analysis\s+Report|Analysis\s+of)[:\s]+(.+?)(?:\*\*)?$",
                trimmed,
                re.IGNORECASE,
            )
            if sub_match:
                subtitle = clean_plain_text(trimmed)
                continue

        # Detect canonical section headers
        matched_sec = None
        for canonical_name, pat in SECTION_CANONICAL_PATTERNS:
            if re.match(pat, trimmed, re.IGNORECASE):
                matched_sec = canonical_name
                break

        if not matched_sec:
            # Detect Markdown section headers (e.g. "### 1. Visually Observed Elements" or "## Section Title")
            if (trimmed.startswith("### ") or trimmed.startswith("## ")) and len(trimmed) < 120:
                matched_sec = clean_plain_text(trimmed)
            else:
                numbered_sec = re.match(r"^(?:#+\s*|\*\*)?(\d+[\.\)]\s+[A-Za-z0-9_\-\s/&]+)(?:\*\*)?$", trimmed)
                if numbered_sec and len(trimmed) < 120:
                    matched_sec = clean_plain_text(trimmed)

        if matched_sec:
            if current_sec_title:
                flush_lines_to_section(current_sec_title, current_lines)
            current_sec_title = matched_sec
            current_lines = []
        else:
            if current_sec_title:
                current_lines.append(line)

    if current_sec_title and current_lines:
        flush_lines_to_section(current_sec_title, current_lines)

    # Order canonical sections strictly once
    ordered_canonical = [
        "1. Executive Summary",
        "A. Executive Summary",
        "2. Key Findings",
        "B. Key Findings",
        "3. Detailed Analysis",
        "C. Detailed Analysis",
        "D. Important Notes / Exceptions",
        "Important Notes / Exceptions",
        "4. Evidence / Page References",
        "E. Evidence / Page References",
        "5. Evidence / Page References",
        "F. Analysis / Interpretation",
        "Analysis / Interpretation",
        "G. Recommendations",
        "Recommendations",
        "5. Conclusion",
    ]

    sections: list[ReportSection] = []
    seen_sections = set()
    for c_title in ordered_canonical:
        if c_title in sections_dict and sections_dict[c_title]:
            sections.append(ReportSection(title=c_title, elements=sections_dict[c_title]))
            seen_sections.add(c_title)

    # Include any dynamic/custom sections (e.g. Visually Observed Elements, Architecture, etc.)
    for sec_title, elements in sections_dict.items():
        if sec_title not in seen_sections and sec_title != "Sources" and elements:
            sections.append(ReportSection(title=sec_title, elements=elements))
            seen_sections.add(sec_title)

    # Collect Sources
    if "Sources" in sections_dict and sections_dict["Sources"]:
        for el in sections_dict["Sources"]:
            sources.append(el.clean_text)

    # Fallback for completely unstructured content
    if not sections:
        raw_paras = [p.strip() for p in content.split("\n\n") if p.strip()]
        if raw_paras:
            sections.append(
                ReportSection(
                    title="1. Executive Summary",
                    elements=[
                        ReportElement(
                            kind="paragraph",
                            raw_text=raw_paras[0],
                            clean_text=clean_plain_text(raw_paras[0]),
                            html_text=markdown_to_reportlab_html(raw_paras[0]),
                            docx_runs=parse_markdown_runs(raw_paras[0]),
                        )
                    ],
                )
            )
            if len(raw_paras) > 1:
                detail_elements = [
                    ReportElement(
                        kind="paragraph",
                        raw_text=p,
                        clean_text=clean_plain_text(p),
                        html_text=markdown_to_reportlab_html(p),
                        docx_runs=parse_markdown_runs(p),
                    )
                    for p in raw_paras[1:]
                ]
                sections.append(ReportSection(title="3. Detailed Analysis", elements=detail_elements))

    formatted_ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return ParsedReport(
        title=title,
        subtitle=subtitle,
        document_name=document_name,
        organization="MRPL Sovereign AI Workbench",
        environment="On-Premise Sovereign Infrastructure (Air-Gapped)",
        timestamp=formatted_ts,
        notice="AI-GENERATED — REQUIRES HUMAN REVIEW",
        sections=sections,
        sources=sources,
    )


def _build_structured_sections(content: str) -> list[tuple[str, str]]:
    """Backward-compatible helper returning list of (section_title, section_body) tuples."""
    report = parse_analysis_report(content, document_name="Document")
    return [(s.title, "\n".join(e.clean_text for e in s.elements)) for s in report.sections]


def render_report_to_pdf(report: ParsedReport, output_path: Path) -> Path:
    """Render a ParsedReport object into a high-fidelity PDF via ReportLab."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0A2540"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=8,
    )

    meta_label_style = ParagraphStyle(
        "ReportMetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
    )

    meta_value_style = ParagraphStyle(
        "ReportMetaValue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A202C"),
    )

    notice_style = ParagraphStyle(
        "ReportNotice",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#92400E"),
        alignment=1,
    )

    section_heading_style = ParagraphStyle(
        "ReportSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0A2540"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=16,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    subheading_style = ParagraphStyle(
        "ReportSubheading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True,
    )

    story = []

    # 1. Title & Subtitle
    story.append(Paragraph(html.escape(report.title), title_style))
    if report.subtitle:
        story.append(Paragraph(html.escape(report.subtitle), subtitle_style))

    # 2. Metadata Table
    meta_rows = [
        [Paragraph("Document / Image:", meta_label_style), Paragraph(html.escape(report.document_name), meta_value_style)],
    ]
    if report.analysis_id:
        meta_rows.append([Paragraph("Analysis ID:", meta_label_style), Paragraph(html.escape(report.analysis_id), meta_value_style)])
    if report.image_hash:
        disp_hash = report.image_hash[:16] + "..." + report.image_hash[-8:] if len(report.image_hash) > 24 else report.image_hash
        meta_rows.append([Paragraph("Image Hash (SHA-256):", meta_label_style), Paragraph(f'<font face="Courier">{html.escape(disp_hash)}</font>', meta_value_style)])
    if report.request_query:
        meta_rows.append([Paragraph("User Request:", meta_label_style), Paragraph(html.escape(report.request_query[:160]), meta_value_style)])
    if report.analysis_source:
        meta_rows.append([Paragraph("Analysis Source:", meta_label_style), Paragraph(html.escape(report.analysis_source.upper()), meta_value_style)])
    if report.verification_status:
        meta_rows.append([Paragraph("Verification Status:", meta_label_style), Paragraph(html.escape(report.verification_status), meta_value_style)])

    meta_rows.extend([
        [Paragraph("Generated by:", meta_label_style), Paragraph(html.escape(report.organization), meta_value_style)],
        [Paragraph("Environment:", meta_label_style), Paragraph(html.escape(report.environment), meta_value_style)],
        [Paragraph("Timestamp:", meta_label_style), Paragraph(html.escape(report.timestamp), meta_value_style)],
    ])

    meta_table = Table(meta_rows, colWidths=[120, 384])
    meta_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 3. Security Alert Banner
    notice_table = Table(
        [[Paragraph(report.notice, notice_style)]],
        colWidths=[504],
    )
    notice_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F59E0B")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(notice_table)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceBefore=10, spaceAfter=8))

    # 4. Embedded Visual Input (if image provided)
    if report.image_bytes or report.image_path:
        try:
            raw_img = report.image_bytes
            if not raw_img and report.image_path and Path(report.image_path).exists():
                with open(report.image_path, "rb") as f_img:
                    raw_img = f_img.read()
            if raw_img:
                with PILImage.open(io.BytesIO(raw_img)) as pil_img:
                    w_px, h_px = pil_img.size
                max_w = 460.0
                max_h = 240.0
                aspect = w_px / float(h_px) if h_px > 0 else 1.0
                if w_px > max_w:
                    target_w = max_w
                    target_h = max_w / aspect
                else:
                    target_w = float(w_px)
                    target_h = float(h_px)
                if target_h > max_h:
                    target_h = max_h
                    target_w = max_h * aspect

                story.append(Paragraph("<b>Input Diagram / Schematic Evidence</b>", subheading_style))
                story.append(Spacer(1, 4))
                story.append(RLImage(io.BytesIO(raw_img), width=target_w, height=target_h))
                caption_text = f"<i>Figure 1: Analyzed Visual Input — {html.escape(report.document_name)}</i>"
                if report.image_hash:
                    caption_text += f" &nbsp;[SHA-256: <code>{html.escape(report.image_hash[:16])}...</code>]"
                caption_style = ParagraphStyle(
                    "ImageCaption",
                    parent=styles["Normal"],
                    fontName="Helvetica",
                    fontSize=8,
                    textColor=colors.HexColor("#64748B"),
                    alignment=1,
                    spaceBefore=4,
                    spaceAfter=8,
                )
                story.append(Paragraph(caption_text, caption_style))
                story.append(Spacer(1, 4))
        except Exception as img_err:
            logger.warning("Could not embed image into PDF: %s", img_err)
            story.append(Paragraph(f"<i>[Warning: Uploaded image could not be embedded into PDF report: {html.escape(str(img_err))}]</i>", body_style))

    # 5. Canonical and Dynamic Sections
    for sec in report.sections:
        sec_flowables: list[Any] = [Paragraph(html.escape(sec.title), section_heading_style)]
        for elem in sec.elements:
            if elem.kind == "bullet":
                sec_flowables.append(Paragraph(f"&bull;&nbsp;&nbsp;{elem.html_text}", bullet_style))
            elif elem.kind == "numbered":
                num_str = f"<b>{elem.number}.</b>&nbsp;&nbsp;" if elem.number else "&bull;&nbsp;&nbsp;"
                sec_flowables.append(Paragraph(f"{num_str}{elem.html_text}", bullet_style))
            elif elem.kind == "subheading":
                sec_flowables.append(Paragraph(elem.html_text, subheading_style))
            else:
                sec_flowables.append(Paragraph(elem.html_text, body_style))
        sec_flowables.append(Spacer(1, 4))
        story.append(KeepTogether(sec_flowables[:2]))
        story.extend(sec_flowables[2:])

    # 5. Sources Block
    if report.sources:
        sources_flowables = [
            Paragraph("Sources", section_heading_style),
            Spacer(1, 2),
        ]
        for i, src in enumerate(report.sources, 1):
            sources_flowables.append(Paragraph(f"<b>{i}.</b>&nbsp;&nbsp;{html.escape(src)}", bullet_style))
        story.append(KeepTogether(sources_flowables))

    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path


def render_report_to_docx(report: ParsedReport, output_path: Path) -> Path:
    """Render a ParsedReport object into a high-fidelity Word (.docx) document."""
    doc = DocxDocument()

    # Document Core Properties
    doc.core_properties.title = report.title
    doc.core_properties.author = report.organization
    doc.core_properties.subject = f"Analysis of {report.document_name}"

    # Document Title
    h0 = doc.add_heading(report.title, level=0)
    for run in h0.runs:
        run.font.color.rgb = RGBColor(10, 37, 64)

    # Subtitle
    if report.subtitle:
        h2 = doc.add_heading(report.subtitle, level=2)
        for run in h2.runs:
            run.font.color.rgb = RGBColor(37, 99, 235)

    # Metadata lines
    doc.add_paragraph(f"Document: {report.document_name}")
    doc.add_paragraph(f"Generated by: {report.organization}")
    doc.add_paragraph(f"Environment: {report.environment}")
    doc.add_paragraph(f"Timestamp: {report.timestamp}")

    # Security Notice
    p_notice = doc.add_paragraph()
    r_notice = p_notice.add_run(report.notice)
    r_notice.bold = True
    r_notice.font.color.rgb = RGBColor(180, 83, 9)

    # Sections
    for sec in report.sections:
        h1 = doc.add_heading(sec.title, level=1)
        for run in h1.runs:
            run.font.color.rgb = RGBColor(10, 37, 64)
        for elem in sec.elements:
            if elem.kind == "bullet":
                p = doc.add_paragraph(style="List Bullet")
                for text, is_bold, is_italic in elem.docx_runs:
                    run = p.add_run(text)
                    run.bold = is_bold
                    run.italic = is_italic
            elif elem.kind == "numbered":
                p = doc.add_paragraph(style="List Bullet")
                num_prefix = f"{elem.number}. " if elem.number else ""
                if num_prefix:
                    r_num = p.add_run(num_prefix)
                    r_num.bold = True
                for text, is_bold, is_italic in elem.docx_runs:
                    run = p.add_run(text)
                    run.bold = is_bold
                    run.italic = is_italic
            elif elem.kind == "subheading":
                p = doc.add_paragraph()
                r_sub = p.add_run(elem.clean_text)
                r_sub.bold = True
                r_sub.font.color.rgb = RGBColor(30, 58, 138)
            else:
                p = doc.add_paragraph()
                for text, is_bold, is_italic in elem.docx_runs:
                    run = p.add_run(text)
                    run.bold = is_bold
                    run.italic = is_italic

    # Sources Block
    if report.sources:
        h_src = doc.add_heading("Sources", level=1)
        for run in h_src.runs:
            run.font.color.rgb = RGBColor(10, 37, 64)
        for i, src in enumerate(report.sources, 1):
            p_s = doc.add_paragraph(style="List Bullet")
            r_idx = p_s.add_run(f"{i}. ")
            r_idx.bold = True
            p_s.add_run(src)

    doc.save(str(output_path))
    return output_path


def generate_pdf(
    title: str = "DOCUMENT ANALYSIS REPORT",
    content: str = "",
    document_name: str | None = None,
    filename: str | None = None,
    metadata: dict | None = None,
    image_bytes: bytes | None = None,
    image_path: str | None = None,
    report_data: dict | None = None,
) -> dict:
    """
    Generate a professional technical PDF report using ReportLab.
    Strictly parses existing analysis without duplicating sections, eliminates raw Markdown,
    embeds image evidence if available, attaches analysis metadata,
    and verifies physical file existence and readability before returning success.
    """
    _ensure_output_dir()

    meta = dict(metadata or {})
    if report_data:
        if "title" in report_data and (not title or title == "DOCUMENT ANALYSIS REPORT"):
            title = report_data["title"]
        if "filename" in report_data and not document_name:
            document_name = report_data["filename"]
        if "document_name" in report_data and not document_name:
            document_name = report_data["document_name"]
        for k in ("analysis_id", "image_hash", "source", "verification_status", "request", "request_text"):
            if k in report_data and k not in meta:
                meta[k] = report_data[k]
        if not content and "sections" in report_data:
            sec_blocks = []
            if "summary" in report_data and report_data["summary"]:
                sec_blocks.append(f"### 1. Executive Summary\n{report_data['summary']}")
            for sec in report_data["sections"]:
                if isinstance(sec, dict):
                    stitle = sec.get("title", "Section")
                    sbody = sec.get("content", sec.get("body", ""))
                    sec_blocks.append(f"### {stitle}\n{sbody}")
            content = "\n\n".join(sec_blocks)

    doc_label = document_name or (meta.get("document_name") if meta else None) or "Document"
    if not filename:
        clean_doc = _safe_filename(doc_label, ext="")
        fname = f"{clean_doc}_Analysis.pdf"
    else:
        fname = _safe_filename(filename, ext=".pdf")

    output_path = _resolve_output_path(fname)

    # 1. Parse content into canonical report structure
    ts = (meta.get("timestamp") if meta else None) or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = parse_analysis_report(content=content, document_name=doc_label, title=title, timestamp=ts)

    # Attach dynamic metadata & image evidence
    report.analysis_id = str(meta.get("analysis_id", ""))
    report.image_hash = str(meta.get("image_hash", ""))
    report.analysis_source = str(meta.get("source", ""))
    report.verification_status = str(meta.get("verification_status", ""))
    report.request_query = str(meta.get("request") or meta.get("request_text", ""))
    report.image_bytes = image_bytes
    report.image_path = image_path

    # 2. Render to PDF
    try:
        render_report_to_pdf(report, output_path)
    except Exception as exc:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"PDF build failed: {exc}") from exc

    # 3. Strict File Existence & Integrity Verification
    is_valid, err_msg = verify_pdf(output_path)
    if not is_valid:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"PDF verification failed: {err_msg}")

    file_size = output_path.stat().st_size
    logger.info("Successfully generated and verified PDF: %s (%d bytes)", fname, file_size)

    return {
        "path": str(output_path),
        "filename": fname,
        "file_size_bytes": file_size,
    }


def generate_docx(
    title: str = "DOCUMENT ANALYSIS REPORT",
    content: str = "",
    document_name: str | None = None,
    filename: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Generate a professional Word document (.docx) using python-docx.
    Applies the exact same canonical parsed report structure as the PDF generator.
    """
    _ensure_output_dir()

    doc_label = document_name or (metadata.get("document_name") if metadata else None) or "Document"
    if not filename:
        clean_doc = _safe_filename(doc_label, ext="")
        fname = f"{clean_doc}_Analysis.docx"
    else:
        fname = _safe_filename(filename, ext=".docx")

    output_path = _resolve_output_path(fname)

    # 1. Parse content into canonical report structure
    ts = (metadata.get("timestamp") if metadata else None) or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = parse_analysis_report(content=content, document_name=doc_label, title=title, timestamp=ts)

    # 2. Render to DOCX
    try:
        render_report_to_docx(report, output_path)
    except Exception as exc:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"DOCX build failed: {exc}") from exc

    # 3. Strict File Existence & Integrity Verification
    is_valid, err_msg = verify_docx(output_path)
    if not is_valid:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"DOCX verification failed: {err_msg}")

    file_size = output_path.stat().st_size
    logger.info("Successfully generated and verified DOCX: %s (%d bytes)", fname, file_size)

    return {
        "path": str(output_path),
        "filename": fname,
        "file_size_bytes": file_size,
    }


def generate_xlsx(
    title: str,
    data: list[list],
    metadata: dict | None = None,
    filename: str | None = None,
    headers: list[str] | None = None,
) -> dict:
    """Generate an Excel workbook (.xlsx)."""
    _ensure_output_dir()
    fname = filename or f"{_safe_filename(title, ext='.xlsx')}"
    output_path = _resolve_output_path(fname)

    wb = Workbook()
    ws = wb.active
    if ws is not None:
        ws.title = title[:31]
        ws.append(["AI-GENERATED — REQUIRES HUMAN REVIEW"])
        ws.append([])

        if headers:
            ws.append(headers)

        for row in data:
            ws.append(row)

    wb.save(str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Spreadsheet generation failed: {output_path} not created")

    file_size = output_path.stat().st_size
    logger.info("Generated XLSX: %s (%d bytes)", fname, file_size)

    return {
        "path": str(output_path),
        "filename": fname,
        "file_size_bytes": file_size,
    }


def generate_pptx(
    title: str,
    slides: list[dict],
    metadata: dict | None = None,
    filename: str | None = None,
) -> dict:
    """Generate a PowerPoint presentation (.pptx)."""
    _ensure_output_dir()
    fname = filename or f"{_safe_filename(title, ext='.pptx')}"
    output_path = _resolve_output_path(fname)

    prs = Presentation()

    title_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_layout)
    if slide.shapes.title is not None:
        slide.shapes.title.text = title
    if len(slide.placeholders) > 1:
        setattr(slide.placeholders[1], "text", "AI-GENERATED — REQUIRES HUMAN REVIEW")

    content_layout = prs.slide_layouts[1]
    for slide_data in slides:
        s = prs.slides.add_slide(content_layout)
        if s.shapes.title is not None:
            s.shapes.title.text = slide_data.get("title", "")
        if len(s.placeholders) > 1:
            # Check for bullet_points list or content text
            bullet_points = slide_data.get("bullet_points")
            if bullet_points and isinstance(bullet_points, list):
                setattr(s.placeholders[1], "text", "\n".join(f"• {bp}" for bp in bullet_points))
            else:
                setattr(s.placeholders[1], "text", slide_data.get("content", ""))

    prs.save(str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"Presentation generation failed: {output_path} not created")

    file_size = output_path.stat().st_size
    logger.info("Generated PPTX: %s (%d bytes)", fname, file_size)

    return {
        "path": str(output_path),
        "filename": fname,
        "file_size_bytes": file_size,
    }
