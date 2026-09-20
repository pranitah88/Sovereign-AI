"""
MRPL SAFE KNOWLEDGE BASE AUDITOR
================================

Purpose:
    Safely audit and organize an MRPL RAG knowledge base.

IMPORTANT:
    This script DOES NOT DELETE files.
    This script DOES NOT automatically remove potentially useful documents.

It checks:

1. Exact duplicate files using SHA-256
2. Empty / 0-byte files
3. File signatures / magic bytes
4. PDFs that are suspiciously tiny
5. PDFs that PyMuPDF cannot open
6. PDFs with text
7. PDFs with images (possible scanned documents)
8. PDFs with neither useful text nor images
9. Basic content quality
10. Files likely to be HTML/error pages despite .pdf extension
11. Creates a detailed JSON audit report

It does NOT:
    - delete files
    - move files
    - decide relevance based only on text length
    - delete scanned PDFs
    - delete Hindi/non-English files
    - delete certificates
    - delete reports
    - delete policies
    - delete environmental documents
    - delete financial documents

The goal is to identify files for HUMAN REVIEW.

Recommended folder:

    D:\\MRPL-Sovereign-AI\\merged_knowledge_base

Run:

    python clean_kb_safe.py
"""

import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(r"D:\MRPL-Sovereign-AI\merged_knowledge_base")

REPORT_FILE = BASE_DIR / "safe_audit_report.json"

# Files smaller than this are suspicious.
# We DO NOT delete them.
SUSPICIOUS_PDF_SIZE = 10 * 1024  # 10 KB

# Text length below this does NOT automatically mean bad.
# It is only used as a warning.
LOW_TEXT_WARNING = 50

# Supported document extensions
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".ppt",
    ".pptx",
    ".txt",
    ".json",
}


# ======================================================================
# SHA256
# ======================================================================

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hash without loading entire file into RAM."""

    h = hashlib.sha256()

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return ""


# ======================================================================
# FILE SIZE
# ======================================================================

def human_size(size: int) -> str:

    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"

    return f"{size / (1024 * 1024 * 1024):.1f} GB"


# ======================================================================
# FILE SIGNATURE
# ======================================================================

def get_signature(path: Path):

    try:
        with open(path, "rb") as f:
            return f.read(16)

    except Exception:
        return b""


def signature_status(path: Path):

    ext = path.suffix.lower()

    try:
        head = get_signature(path)

        if not head:
            return "UNREADABLE"

        # PDF
        if ext == ".pdf":

            if head.startswith(b"%PDF"):
                return "VALID_PDF_SIGNATURE"

            # Sometimes a server returns HTML instead of PDF.
            stripped = head.lstrip().lower()

            if (
                stripped.startswith(b"<html")
                or stripped.startswith(b"<!doctype")
                or stripped.startswith(b"<")
            ):
                return "LIKELY_HTML_OR_ERROR_PAGE"

            return "INVALID_PDF_SIGNATURE"

        # ZIP-based Office files
        if ext in {".docx", ".xlsx", ".pptx"}:

            if head.startswith(b"PK"):
                return "VALID_ZIP_OFFICE_SIGNATURE"

            return "INVALID_OFFICE_SIGNATURE"

        # Old Microsoft Office
        if ext in {".doc", ".xls", ".ppt"}:

            if head.startswith(b"\xD0\xCF\x11\xE0"):
                return "VALID_OLE_SIGNATURE"

            return "INVALID_OLE_SIGNATURE"

        # Text formats
        if ext in {".txt", ".csv", ".json"}:
            return "TEXT_FILE"

        return "NOT_CHECKED"

    except Exception as e:

        return f"SIGNATURE_CHECK_ERROR: {e}"


# ======================================================================
# PDF ANALYSIS
# ======================================================================

def analyze_pdf(path: Path):

    result = {
        "openable": False,
        "page_count": 0,
        "text_chars": 0,
        "image_count": 0,
        "pages_with_text": 0,
        "pages_with_images": 0,
        "scanned_possible": False,
        "quality": "UNKNOWN",
        "error": None,
    }

    try:

        import fitz

    except ImportError:

        result["error"] = (
            "PyMuPDF not installed. "
            "Run: pip install pymupdf"
        )

        return result

    try:

        doc = fitz.open(path)

        result["openable"] = True
        result["page_count"] = len(doc)

        total_text = 0
        total_images = 0
        pages_text = 0
        pages_images = 0

        for page in doc:

            try:
                text = page.get_text("text") or ""

                text_len = len(text.strip())

                if text_len > 0:
                    pages_text += 1

                total_text += text_len

                images = page.get_images(full=True)

                image_count = len(images)

                if image_count > 0:
                    pages_images += 1

                total_images += image_count

            except Exception:
                continue

        doc.close()

        result["text_chars"] = total_text
        result["image_count"] = total_images
        result["pages_with_text"] = pages_text
        result["pages_with_images"] = pages_images

        # --------------------------------------------------------------
        # VERY IMPORTANT:
        #
        # A scanned PDF can have almost zero text but still contain
        # valuable information as images.
        #
        # Therefore we DO NOT call it useless.
        # --------------------------------------------------------------

        if total_text >= LOW_TEXT_WARNING:

            result["quality"] = "TEXT_CONTENT_AVAILABLE"

        elif total_images > 0:

            result["scanned_possible"] = True
            result["quality"] = "POSSIBLE_SCANNED_DOCUMENT"

        elif result["page_count"] > 0:

            result["quality"] = "LOW_CONTENT_PDF"

        else:

            result["quality"] = "EMPTY_PDF"

        return result

    except Exception as e:

        result["error"] = str(e)
        result["quality"] = "PDF_OPEN_FAILED"

        return result


# ======================================================================
# TEXT FILE ANALYSIS
# ======================================================================

def analyze_text_file(path: Path):

    result = {
        "text_chars": 0,
        "quality": "UNKNOWN",
        "error": None,
    }

    try:

        # Try UTF-8 first
        try:

            text = path.read_text(
                encoding="utf-8",
                errors="replace"
            )

        except Exception:

            text = path.read_text(
                encoding="utf-16",
                errors="replace"
            )

        stripped = text.strip()

        result["text_chars"] = len(stripped)

        if len(stripped) == 0:

            result["quality"] = "EMPTY_TEXT_FILE"

        elif len(stripped) < LOW_TEXT_WARNING:

            result["quality"] = "VERY_SHORT_TEXT"

        else:

            result["quality"] = "TEXT_CONTENT_AVAILABLE"

        return result

    except Exception as e:

        result["error"] = str(e)
        result["quality"] = "TEXT_READ_FAILED"

        return result


# ======================================================================
# OFFICE DOCUMENT ANALYSIS
# ======================================================================

def analyze_office_file(path: Path):

    ext = path.suffix.lower()

    result = {
        "text_chars": 0,
        "quality": "UNKNOWN",
        "error": None,
    }

    try:

        if ext == ".docx":

            from docx import Document

            doc = Document(path)

            text = "\n".join(
                p.text for p in doc.paragraphs
            )

            result["text_chars"] = len(text.strip())

        elif ext == ".xlsx":

            import openpyxl

            wb = openpyxl.load_workbook(
                path,
                read_only=True,
                data_only=True
            )

            total = 0

            for ws in wb.worksheets:

                for row in ws.iter_rows(values_only=True):

                    for value in row:

                        if value is not None:

                            total += len(str(value).strip())

            result["text_chars"] = total

            wb.close()

        elif ext == ".pptx":

            from pptx import Presentation

            prs = Presentation(path)

            total = 0

            for slide in prs.slides:

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        total += len(shape.text.strip())

            result["text_chars"] = total

        else:

            result["quality"] = "NOT_ANALYZED"

            return result

        if result["text_chars"] >= LOW_TEXT_WARNING:

            result["quality"] = "TEXT_CONTENT_AVAILABLE"

        else:

            result["quality"] = "LOW_TEXT_CONTENT"

        return result

    except Exception as e:

        result["error"] = str(e)
        result["quality"] = "OFFICE_READ_FAILED"

        return result


# ======================================================================
# SINGLE FILE AUDIT
# ======================================================================

def audit_file(path: Path):

    try:

        size = path.stat().st_size

    except Exception as e:

        return {
            "path": str(path),
            "filename": path.name,
            "status": "STAT_FAILED",
            "error": str(e),
        }

    record = {

        "path": str(path),

        "filename": path.name,

        "extension": path.suffix.lower(),

        "size_bytes": size,

        "size_human": human_size(size),

        "sha256": "",

        "signature": None,

        "category": None,

        "warnings": [],

        "recommendation": "KEEP_UNTIL_MANUALLY_REVIEWED",

    }

    # --------------------------------------------------------------
    # SHA256
    # --------------------------------------------------------------

    record["sha256"] = sha256_file(path)

    # --------------------------------------------------------------
    # Empty file
    # --------------------------------------------------------------

    if size == 0:

        record["category"] = "EMPTY_FILE"

        record["warnings"].append(
            "File is 0 bytes."
        )

        return record

    # --------------------------------------------------------------
    # Signature
    # --------------------------------------------------------------

    record["signature"] = signature_status(path)

    # --------------------------------------------------------------
    # PDF
    # --------------------------------------------------------------

    if path.suffix.lower() == ".pdf":

        pdf = analyze_pdf(path)

        record["pdf"] = pdf

        # Suspiciously tiny PDF
        if size < SUSPICIOUS_PDF_SIZE:

            record["warnings"].append(
                f"PDF is unusually small: {human_size(size)}"
            )

        # Invalid signature
        if record["signature"] != "VALID_PDF_SIGNATURE":

            record["category"] = "INVALID_PDF"

            record["warnings"].append(
                "File extension says PDF but PDF signature is invalid."
            )

            return record

        # Cannot open
        if not pdf["openable"]:

            record["category"] = "PDF_OPEN_FAILED"

            record["warnings"].append(
                "PyMuPDF could not open this PDF."
            )

            return record

        # Scanned document
        if pdf["scanned_possible"]:

            record["category"] = "POSSIBLE_SCANNED_PDF"

            record["warnings"].append(
                "PDF has little text but contains images. "
                "It may be a scanned document and should NOT be deleted."
            )

            return record

        # Normal text PDF
        if pdf["text_chars"] >= LOW_TEXT_WARNING:

            record["category"] = "GOOD_TEXT_PDF"

            return record

        # Openable but almost no text/images
        record["category"] = "LOW_CONTENT_PDF"

        record["warnings"].append(
            "PDF opens successfully but has little/no extractable text "
            "and no detected page images."
        )

        return record

    # --------------------------------------------------------------
    # TXT / CSV / JSON
    # --------------------------------------------------------------

    if path.suffix.lower() in {".txt", ".csv", ".json"}:

        text_info = analyze_text_file(path)

        record["text"] = text_info

        if text_info["quality"] == "EMPTY_TEXT_FILE":

            record["category"] = "EMPTY_TEXT_FILE"

            record["warnings"].append(
                "Text file contains no usable text."
            )

        elif text_info["quality"] == "VERY_SHORT_TEXT":

            record["category"] = "SHORT_TEXT_FILE"

            record["warnings"].append(
                "Text file is very short. Manual review recommended."
            )

        else:

            record["category"] = "GOOD_TEXT_FILE"

        return record

    # --------------------------------------------------------------
    # DOCX / XLSX / PPTX
    # --------------------------------------------------------------

    if path.suffix.lower() in {
        ".docx",
        ".xlsx",
        ".pptx",
    }:

        office = analyze_office_file(path)

        record["office"] = office

        if office["quality"] == "TEXT_CONTENT_AVAILABLE":

            record["category"] = "GOOD_OFFICE_DOCUMENT"

        else:

            record["category"] = "LOW_CONTENT_OFFICE_DOCUMENT"

            record["warnings"].append(
                "Office document contains little extractable text."
            )

        return record

    # --------------------------------------------------------------
    # Other files
    # --------------------------------------------------------------

    record["category"] = "OTHER_FILE"

    record["warnings"].append(
        "File type is not deeply analyzed by this script."
    )

    return record


# ======================================================================
# DUPLICATE ANALYSIS
# ======================================================================

def find_duplicates(records):

    groups = defaultdict(list)

    for record in records:

        sha = record.get("sha256")

        if sha:
            groups[sha].append(record["path"])

    duplicate_sets = []

    duplicate_file_count = 0

    for sha, paths in groups.items():

        if len(paths) > 1:

            duplicate_sets.append({

                "sha256": sha,

                "files": paths,

                "count": len(paths),

            })

            duplicate_file_count += len(paths) - 1

    return duplicate_sets, duplicate_file_count


# ======================================================================
# IMPORTANT MRPL DOCUMENT CLASSIFICATION
# ======================================================================

def classify_mrpl_importance(record):

    """
    This is intentionally conservative.

    It does NOT delete anything.

    It only provides a REVIEW PRIORITY based on filename/category.

    High priority examples:
        annual reports
        financial results
        policies
        certificates
        environmental reports
        safety/HSE
        manufacturing/refining
        company information

    Unknown documents are NOT marked irrelevant.
    """

    filename = record["filename"].lower()

    important_keywords = {

        "annual": "Annual report / annual document",

        "financial": "Financial information",

        "result": "Financial/result information",

        "board": "Board information",

        "environment": "Environmental information",

        "ec_compliance": "Environmental compliance",

        "consent": "Environmental/regulatory consent",

        "green": "Environmental information",

        "safety": "Safety/HSE information",

        "hse": "Safety/HSE information",

        "policy": "Company policy",

        "certificate": "Certification",

        "iso": "ISO/certification",

        "manufacturing": "Manufacturing/refining",

        "refining": "Manufacturing/refining",

        "petrochemical": "Petrochemical information",

        "product": "Product information",

        "msds": "Chemical/product safety information",

        "xylol": "Chemical/product safety information",

        "toluene": "Chemical/product safety information",

        "xylenes": "Chemical/product safety information",

        "csr": "CSR information",

        "skill": "Skill development / CSR",

        "report": "Report",

        "vigilance": "Vigilance/compliance",

        "security": "Security policy",

        "retention": "Record retention policy",

        "inspection": "Inspection information",

        "compliance": "Compliance information",

        "brochure": "Company/product information",

        "mrpl": "MRPL-specific document",

    }

    matches = []

    for keyword, reason in important_keywords.items():

        if keyword in filename:

            matches.append(reason)

    if matches:

        record["importance"] = "HIGH_REVIEW_PRIORITY"
        record["importance_reasons"] = sorted(set(matches))

    else:

        record["importance"] = "UNKNOWN_REVIEW_PRIORITY"
        record["importance_reasons"] = [
            "Filename alone is insufficient to determine relevance."
        ]

    return record


# ======================================================================
# MAIN AUDIT
# ======================================================================

def main():

    print("=" * 70)
    print("MRPL SAFE KNOWLEDGE BASE AUDIT")
    print("=" * 70)

    print()
    print("Knowledge Base:")
    print(BASE_DIR)

    print()
    print("IMPORTANT:")
    print("  NOTHING WILL BE DELETED")
    print("  NOTHING WILL BE MOVED")
    print("  NOTHING WILL BE OVERWRITTEN")

    print()
    print("=" * 70)

    if not BASE_DIR.exists():

        print()
        print("ERROR:")
        print("Knowledge base does not exist:")
        print(BASE_DIR)
        return

    # --------------------------------------------------------------
    # Find files
    # --------------------------------------------------------------

    files = []

    for path in BASE_DIR.rglob("*"):

        if not path.is_file():
            continue

        # Don't audit our own reports
        if path.name in {
            "safe_audit_report.json",
            "audit_report.json",
        }:
            continue

        files.append(path)

    print()
    print(f"Files found: {len(files)}")

    print()
    print("=" * 70)
    print("AUDITING FILES")
    print("=" * 70)

    records = []

    for index, path in enumerate(files, start=1):

        print(
            f"[{index}/{len(files)}] "
            f"{path.name}"
        )

        record = audit_file(path)

        record = classify_mrpl_importance(record)

        records.append(record)

    # --------------------------------------------------------------
    # Duplicate analysis
    # --------------------------------------------------------------

    duplicate_sets, duplicate_file_count = find_duplicates(records)

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    stats = {

        "total_files": len(records),

        "duplicate_sets": len(duplicate_sets),

        "duplicate_files_that_can_be_removed": duplicate_file_count,

        "empty_files": 0,

        "invalid_pdfs": 0,

        "pdf_open_failures": 0,

        "suspicious_small_pdfs": 0,

        "good_text_pdfs": 0,

        "possible_scanned_pdfs": 0,

        "low_content_pdfs": 0,

        "good_text_files": 0,

        "short_text_files": 0,

        "high_review_priority": 0,

        "unknown_review_priority": 0,

    }

    for record in records:

        category = record.get("category")

        if category == "EMPTY_FILE":
            stats["empty_files"] += 1

        elif category == "INVALID_PDF":
            stats["invalid_pdfs"] += 1

        elif category == "PDF_OPEN_FAILED":
            stats["pdf_open_failures"] += 1

        elif category == "GOOD_TEXT_PDF":
            stats["good_text_pdfs"] += 1

        elif category == "POSSIBLE_SCANNED_PDF":
            stats["possible_scanned_pdfs"] += 1

        elif category == "LOW_CONTENT_PDF":
            stats["low_content_pdfs"] += 1

        elif category == "GOOD_TEXT_FILE":
            stats["good_text_files"] += 1

        elif category == "SHORT_TEXT_FILE":
            stats["short_text_files"] += 1

        if record.get("size_bytes", 0) < SUSPICIOUS_PDF_SIZE:
            if record.get("extension") == ".pdf":
                stats["suspicious_small_pdfs"] += 1

        if record.get("importance") == "HIGH_REVIEW_PRIORITY":
            stats["high_review_priority"] += 1

        elif record.get("importance") == "UNKNOWN_REVIEW_PRIORITY":
            stats["unknown_review_priority"] += 1

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    report = {

        "generated_at": datetime.now().isoformat(),

        "base_directory": str(BASE_DIR),

        "safe_mode": True,

        "deletion_performed": False,

        "movement_performed": False,

        "statistics": stats,

        "duplicate_sets": duplicate_sets,

        "files": records,

    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print()
    print(f"Files audited:              {stats['total_files']}")
    print(f"Duplicate sets:             {stats['duplicate_sets']}")
    print(
        f"Duplicate copies:           "
        f"{stats['duplicate_files_that_can_be_removed']}"
    )

    print()
    print(f"Empty files:                {stats['empty_files']}")
    print(f"Invalid PDFs:               {stats['invalid_pdfs']}")
    print(f"PDF open failures:          {stats['pdf_open_failures']}")
    print(f"Suspicious small PDFs:      {stats['suspicious_small_pdfs']}")

    print()
    print(f"Good text PDFs:             {stats['good_text_pdfs']}")
    print(
        f"Possible scanned PDFs:      "
        f"{stats['possible_scanned_pdfs']}"
    )
    print(f"Low-content PDFs:           {stats['low_content_pdfs']}")

    print()
    print(f"Good text files:             {stats['good_text_files']}")
    print(f"Short text files:            {stats['short_text_files']}")

    print()
    print(
        f"High review priority:       "
        f"{stats['high_review_priority']}"
    )

    print(
        f"Unknown review priority:    "
        f"{stats['unknown_review_priority']}"
    )

    print()
    print("=" * 70)
    print("REPORT")
    print("=" * 70)

    print(REPORT_FILE)

    print()
    print("=" * 70)
    print("NO FILES WERE DELETED.")
    print("NO FILES WERE MOVED.")
    print("NO FILES WERE MODIFIED.")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()