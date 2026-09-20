import os
import re
import json
import hashlib
import shutil
from pathlib import Path

import fitz  # PyMuPDF


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\MRPL-Sovereign-AI")
KB_DIR = PROJECT_ROOT / "knowledge_base"

DOCUMENTS_DIR = KB_DIR / "documents"
PAGES_DIR = KB_DIR / "pages"

CLEAN_DIR = KB_DIR / "cleaned"
CLEAN_DOCS_DIR = CLEAN_DIR / "documents"
CLEAN_PAGES_DIR = CLEAN_DIR / "pages"

MANIFEST_PATH = CLEAN_DIR / "manifest.json"


# ============================================================
# SUPPORTED DOCUMENT TYPES
# ============================================================

SUPPORTED_DOCUMENTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".ppt",
    ".pptx",
    ".zip",
}


# ============================================================
# BASIC UTILITIES
# ============================================================

def sha256_file(path):
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
        return None


def clean_filename(name):
    name = re.sub(r"[<>:\"/\\|?*]", "_", name)
    name = re.sub(r"\s+", "_", name)

    return name[:180]


def is_real_pdf(path):
    try:
        with open(path, "rb") as f:
            signature = f.read(5)

        if signature != b"%PDF-":
            return False

        doc = fitz.open(path)

        if doc.page_count <= 0:
            doc.close()
            return False

        doc.close()

        return True

    except Exception:
        return False


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_web_text(text):
    if not text:
        return ""

    # Fix common UTF-8/Latin-1 corruption
    replacements = {
        "â†”": "↔",
        "â†’": "→",
        "â€“": "–",
        "â€”": "—",
        "â€œ": '"',
        "â€": '"',
        "â€˜": "'",
        "â€™": "'",
        "Â": "",
        "Ã—": "×",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove obvious navigation noise
    noise_patterns = [
        r"^\s*Skip to main content\s*$",
        r"^\s*Screen Reader Access\s*$",
        r"^\s*search\s*$",
        r"^\s*Custom Search\s*$",
        r"^\s*Sort by\s*$",
        r"^\s*Relevance\s*$",
        r"^\s*Date\s*$",
        r"^\s*Promotional image\s*$",
    ]

    lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:
            lines.append("")
            continue

        remove = False

        for pattern in noise_patterns:

            if re.match(pattern, stripped, re.IGNORECASE):
                remove = True
                break

        if not remove:
            lines.append(stripped)

    text = "\n".join(lines)

    # Remove excessive blank lines again
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(pdf_path):

    try:

        doc = fitz.open(pdf_path)

        pages = []

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            if text:
                text = clean_web_text(text)

                if text:
                    pages.append(
                        f"\n--- PAGE {page_number} ---\n{text}"
                    )

        doc.close()

        final_text = "\n".join(pages)

        return final_text.strip()

    except Exception as e:

        print(f"    PDF extraction failed: {e}")

        return ""


# ============================================================
# FIND ALL FILES
# ============================================================

def collect_files():

    documents = []
    pages = []

    if DOCUMENTS_DIR.exists():

        for path in DOCUMENTS_DIR.rglob("*"):

            if path.is_file():

                if path.suffix.lower() in SUPPORTED_DOCUMENTS:

                    documents.append(path)

    if PAGES_DIR.exists():

        for path in PAGES_DIR.rglob("*.txt"):

            if path.is_file():

                pages.append(path)

    return documents, pages


# ============================================================
# CLEAN DOCUMENT DATASET
# ============================================================

def process_documents(documents):

    print()
    print("=" * 70)
    print("CLEANING DOCUMENTS")
    print("=" * 70)

    CLEAN_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()

    manifest = []

    stats = {
        "total": 0,
        "valid_pdf": 0,
        "invalid_pdf": 0,
        "duplicates": 0,
        "copied": 0,
        "text_extracted": 0,
    }

    for source in documents:

        stats["total"] += 1

        print()
        print(f"[{stats['total']}] {source.name}")

        # ----------------------------------------------------
        # HASH
        # ----------------------------------------------------

        file_hash = sha256_file(source)

        if not file_hash:

            print("    FAILED: Could not calculate hash")
            continue

        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        if file_hash in seen_hashes:

            print("    DUPLICATE: skipped")

            stats["duplicates"] += 1

            continue

        seen_hashes.add(file_hash)

        # ----------------------------------------------------
        # PDF VALIDATION
        # ----------------------------------------------------

        if source.suffix.lower() == ".pdf":

            if not is_real_pdf(source):

                print("    INVALID PDF: skipped")

                stats["invalid_pdf"] += 1

                continue

            stats["valid_pdf"] += 1

        # ----------------------------------------------------
        # OUTPUT NAME
        # ----------------------------------------------------

        safe_name = clean_filename(source.name)

        destination = CLEAN_DOCS_DIR / safe_name

        # Prevent filename collisions
        if destination.exists():

            destination = CLEAN_DOCS_DIR / (
                f"{source.stem}_{file_hash[:12]}{source.suffix}"
            )

        # ----------------------------------------------------
        # COPY ORIGINAL DOCUMENT
        # ----------------------------------------------------

        try:

            shutil.copy2(source, destination)

            stats["copied"] += 1

            print(f"    COPIED: {destination.name}")

        except Exception as e:

            print(f"    COPY FAILED: {e}")

            continue

        # ----------------------------------------------------
        # PDF TEXT EXTRACTION
        # ----------------------------------------------------

        extracted_text = ""

        if source.suffix.lower() == ".pdf":

            extracted_text = extract_pdf_text(source)

            if extracted_text:

                stats["text_extracted"] += 1

                text_name = destination.stem + ".txt"

                text_path = CLEAN_DOCS_DIR / text_name

                try:

                    text_path.write_text(
                        extracted_text,
                        encoding="utf-8"
                    )

                except Exception as e:

                    print(f"    TEXT SAVE FAILED: {e}")

        # ----------------------------------------------------
        # MANIFEST ENTRY
        # ----------------------------------------------------

        manifest.append({
            "filename": destination.name,
            "source_filename": source.name,
            "sha256": file_hash,
            "extension": source.suffix.lower(),
            "size_bytes": source.stat().st_size,
            "text_extracted": bool(extracted_text),
        })

    return manifest, stats


# ============================================================
# CLEAN WEB PAGES
# ============================================================

def process_pages():

    print()
    print("=" * 70)
    print("CLEANING WEB PAGES")
    print("=" * 70)

    CLEAN_PAGES_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "saved": 0,
        "empty": 0,
        "duplicates": 0,
    }

    seen_hashes = set()

    manifest = []

    if not PAGES_DIR.exists():

        print("Pages directory does not exist.")

        return manifest, stats

    for source in PAGES_DIR.rglob("*.txt"):

        stats["total"] += 1

        print()
        print(f"[{stats['total']}] {source.name}")

        try:

            raw_text = source.read_text(
                encoding="utf-8",
                errors="replace"
            )

        except Exception as e:

            print(f"    READ FAILED: {e}")

            continue

        cleaned = clean_web_text(raw_text)

        # Ignore useless/empty pages
        if len(cleaned.strip()) < 100:

            print("    TOO LITTLE CONTENT: skipped")

            stats["empty"] += 1

            continue

        content_hash = hashlib.sha256(
            cleaned.encode("utf-8")
        ).hexdigest()

        if content_hash in seen_hashes:

            print("    DUPLICATE PAGE: skipped")

            stats["duplicates"] += 1

            continue

        seen_hashes.add(content_hash)

        destination = CLEAN_PAGES_DIR / source.name

        if destination.exists():

            destination = CLEAN_PAGES_DIR / (
                f"{source.stem}_{content_hash[:12]}.txt"
            )

        try:

            destination.write_text(
                cleaned,
                encoding="utf-8"
            )

            stats["saved"] += 1

            print(f"    SAVED: {destination.name}")

        except Exception as e:

            print(f"    SAVE FAILED: {e}")

            continue

        manifest.append({
            "filename": destination.name,
            "source_filename": source.name,
            "sha256": content_hash,
            "characters": len(cleaned),
        })

    return manifest, stats


# ============================================================
# BUILD MANIFEST
# ============================================================

def save_manifest(document_manifest, page_manifest):

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    data = {

        "project": "MRPL Sovereign AI",

        "description":
            "Cleaned knowledge base generated from MRPL public website data.",

        "documents": document_manifest,

        "web_pages": page_manifest,

        "summary": {
            "documents": len(document_manifest),
            "web_pages": len(page_manifest),
        }
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MRPL KNOWLEDGE BASE CLEANER")
    print("=" * 70)

    print()
    print(f"Original KB:")
    print(KB_DIR)

    print()
    print(f"Clean KB:")
    print(CLEAN_DIR)

    print()
    print("IMPORTANT:")
    print("Original files will NOT be deleted.")
    print("Cleaned files will be placed in:")
    print(CLEAN_DIR)

    # --------------------------------------------------------
    # CHECK KB
    # --------------------------------------------------------

    if not KB_DIR.exists():

        print()
        print("ERROR: knowledge_base does not exist.")

        return

    # --------------------------------------------------------
    # COLLECT
    # --------------------------------------------------------

    documents, pages = collect_files()

    print()
    print("=" * 70)
    print("DATASET FOUND")
    print("=" * 70)

    print(f"Documents: {len(documents)}")
    print(f"Web pages: {len(pages)}")

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    document_manifest, document_stats = process_documents(
        documents
    )

    page_manifest, page_stats = process_pages()

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    save_manifest(
        document_manifest,
        page_manifest
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print()
    print("DOCUMENTS")
    print(f"  Found:              {document_stats['total']}")
    print(f"  Valid PDFs:         {document_stats['valid_pdf']}")
    print(f"  Invalid PDFs:       {document_stats['invalid_pdf']}")
    print(f"  Duplicates:         {document_stats['duplicates']}")
    print(f"  Copied:             {document_stats['copied']}")
    print(f"  PDF text extracted: {document_stats['text_extracted']}")

    print()
    print("WEB PAGES")
    print(f"  Found:              {page_stats['total']}")
    print(f"  Saved:              {page_stats['saved']}")
    print(f"  Too small/empty:    {page_stats['empty']}")
    print(f"  Duplicates:         {page_stats['duplicates']}")

    print()
    print("=" * 70)
    print("CLEAN KB LOCATION")
    print("=" * 70)

    print(CLEAN_DIR)

    print()
    print("Documents:")
    print(CLEAN_DOCS_DIR)

    print()
    print("Pages:")
    print(CLEAN_PAGES_DIR)

    print()
    print("Manifest:")
    print(MANIFEST_PATH)

    print()
    print("=" * 70)
    print("ORIGINAL KNOWLEDGE BASE WAS NOT MODIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()