import os
import re
import json
import time
import hashlib
from pathlib import Path
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag, unquote

import requests
from scrapling.fetchers import StealthyFetcher


# ============================================================
# CONFIGURATION
# ============================================================

START_URL = "https://mrpl.co.in/en/"

PROJECT_ROOT = Path(r"D:\MRPL-Sovereign-AI")
KB_DIR = PROJECT_ROOT / "knowledge_base"
DOC_DIR = KB_DIR / "documents"
TEXT_DIR = KB_DIR / "pages"

MAX_PAGES = 500
MAX_DEPTH = 8

REQUEST_TIMEOUT = 45
BROWSER_TIMEOUT = 60000

CRAWL_DELAY = 1.0
DOCUMENT_RETRIES = 3

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

ALLOWED_DOMAIN = "mrpl.co.in"

DOCUMENT_EXTENSIONS = {
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

# Pages containing these patterns are usually useful for an
# institutional knowledge base.
USEFUL_PAGE_KEYWORDS = [
    "about",
    "overview",
    "history",
    "profile",
    "management",
    "director",
    "board",
    "organization",
    "organisation",
    "role",
    "function",
    "product",
    "products",
    "refinery",
    "petroleum",
    "petrochemical",
    "process",
    "project",
    "environment",
    "environmental",
    "green",
    "safety",
    "occupational",
    "health",
    "quality",
    "iso",
    "policy",
    "information",
    "security",
    "csr",
    "career",
    "recruitment",
    "vendor",
    "supplier",
    "tender",
    "procurement",
    "finance",
    "financial",
    "annual",
    "report",
    "investor",
    "shareholder",
    "notice",
    "circular",
    "certificate",
    "award",
    "technology",
    "innovation",
    "sustainability",
    "energy",
    "water",
    "waste",
    "compliance",
    "legal",
    "contact",
]

# These are generally navigation/noise pages.
SKIP_URL_PATTERNS = [
    "/login",
    "/logout",
    "/captcha",
    "/search",
    "javascript:",
    "mailto:",
    "tel:",
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

KB_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.7",
        "Connection": "keep-alive",
    }
)


# ============================================================
# HELPERS
# ============================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_filename(name: str) -> str:
    name = unquote(name)

    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", name)

    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        name = "document"

    return name[:180]


def normalize_url(url: str) -> str:
    url = urldefrag(url)[0].strip()

    if not url:
        return ""

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return ""

    hostname = parsed.hostname

    if not hostname:
        return ""

    hostname = hostname.lower()

    if hostname == "www.mrpl.co.in":
        hostname = "mrpl.co.in"

    # Normalize domain.
    if hostname != ALLOWED_DOMAIN:
        if not hostname.endswith("." + ALLOWED_DOMAIN):
            return ""

    path = parsed.path or "/"

    # Remove repeated slashes.
    path = re.sub(r"/+", "/", path)

    # Remove trailing slash except root.
    if len(path) > 1:
        path = path.rstrip("/")

    normalized = f"https://{hostname}{path}"

    if parsed.query:
        normalized += "?" + parsed.query

    return normalized


def is_mrpl_url(url: str) -> bool:
    try:
        parsed = urlparse(url)

        hostname = (parsed.hostname or "").lower()

        return (
            hostname == ALLOWED_DOMAIN
            or hostname.endswith("." + ALLOWED_DOMAIN)
        )

    except Exception:
        return False


def is_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()

    decoded = unquote(path)

    return any(
        decoded.endswith(ext)
        for ext in DOCUMENT_EXTENSIONS
    )


def is_probably_useful_page(url: str) -> bool:
    lower = url.lower()

    return any(
        keyword in lower
        for keyword in USEFUL_PAGE_KEYWORDS
    )


def should_skip_url(url: str) -> bool:
    lower = url.lower()

    for pattern in SKIP_URL_PATTERNS:
        if pattern in lower:
            return True

    return False


def get_existing_hashes():
    hashes = set()

    for root, _, files in os.walk(KB_DIR):
        for filename in files:
            path = Path(root) / filename

            try:
                with open(path, "rb") as f:
                    digest = sha256_bytes(f.read())

                hashes.add(digest)

            except Exception:
                pass

    return hashes


def make_unique_path(directory: Path, filename: str) -> Path:
    filename = clean_filename(filename)

    target = directory / filename

    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix

    counter = 2

    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"

        if not candidate.exists():
            return candidate

        counter += 1


# ============================================================
# HTML TEXT EXTRACTION
# ============================================================

def extract_page_text(response) -> str:
    """
    Extract useful visible text from a Scrapling response.

    We deliberately keep this relatively conservative because
    the text itself will later be used by the RAG pipeline.
    """

    try:
        html = response.text
    except Exception:
        return ""

    if not html:
        return ""

    # Remove scripts/styles/noscript.
    html = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        html,
        flags=re.I | re.S,
    )

    html = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        html,
        flags=re.I | re.S,
    )

    html = re.sub(
        r"<noscript\b[^>]*>.*?</noscript>",
        " ",
        html,
        flags=re.I | re.S,
    )

    html = re.sub(
        r"<svg\b[^>]*>.*?</svg>",
        " ",
        html,
        flags=re.I | re.S,
    )

    # Convert common block elements to line breaks.
    html = re.sub(
        r"</(p|div|section|article|li|tr|h1|h2|h3|h4|h5|h6|br|td|th)>",
        "\n",
        html,
        flags=re.I,
    )

    # Remove remaining tags.
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode HTML entities.
    import html as html_module

    text = html_module.unescape(text)

    # Fix common encoding problems.
    replacements = {
        "\xa0": " ",
        "\r": "\n",
        "\t": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize whitespace but preserve line structure.
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    # Remove excessive repeated lines.
    cleaned = []

    previous = None

    for line in lines:
        if line == previous:
            continue

        cleaned.append(line)
        previous = line

    return "\n".join(cleaned).strip()


# ============================================================
# LINK EXTRACTION
# ============================================================

def extract_links(response, current_url):
    links = set()

    try:
        html = response.text
    except Exception:
        return links

    if not html:
        return links

    # href links
    matches = re.findall(
        r'href\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.I,
    )

    # src links can sometimes point directly to documents.
    matches += re.findall(
        r'src\s*=\s*["\']([^"\']+)["\']',
        html,
        flags=re.I,
    )

    for href in matches:
        href = href.strip()

        if not href:
            continue

        if href.startswith("#"):
            continue

        if href.lower().startswith(
            ("javascript:", "mailto:", "tel:")
        ):
            continue

        absolute = urljoin(current_url, href)

        normalized = normalize_url(absolute)

        if not normalized:
            continue

        if not is_mrpl_url(normalized):
            continue

        if should_skip_url(normalized):
            continue

        links.add(normalized)

    return links


# ============================================================
# SAVE WEB PAGE
# ============================================================

def save_page_text(url: str, text: str, existing_hashes):
    if not text or len(text.strip()) < 80:
        return None

    digest = sha256_text(text)

    if digest in existing_hashes:
        print("  TEXT DUPLICATE: skipped")
        return None

    parsed = urlparse(url)

    path_part = parsed.path.strip("/")

    if not path_part:
        path_part = "home"

    # Create a readable base name.
    base = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        path_part,
    ).strip("_")

    if not base:
        base = "page"

    filename = f"{base}_{digest[:12]}.txt"

    target = make_unique_path(TEXT_DIR, filename)

    source_header = (
        "MRPL SOURCE URL:\n"
        f"{url}\n\n"
        "============================================================\n"
        "PAGE CONTENT\n"
        "============================================================\n\n"
    )

    final_text = source_header + text + "\n"

    try:
        target.write_text(
            final_text,
            encoding="utf-8",
        )

        existing_hashes.add(digest)

        print(f"  SAVED TEXT: {target.name}")

        return target

    except Exception as exc:
        print(f"  TEXT SAVE ERROR: {exc}")

        return None


# ============================================================
# DOCUMENT DOWNLOAD
# ============================================================

def get_document_filename(url, content_type):
    parsed = urlparse(url)

    path_name = Path(unquote(parsed.path)).name

    path_name = clean_filename(path_name)

    if path_name and "." in path_name:
        return path_name

    extension = ""

    content_type = content_type.lower()

    if "pdf" in content_type:
        extension = ".pdf"

    elif "word" in content_type:
        extension = ".docx"

    elif "spreadsheet" in content_type or "excel" in content_type:
        extension = ".xlsx"

    elif "presentation" in content_type:
        extension = ".pptx"

    elif "zip" in content_type:
        extension = ".zip"

    if not extension:
        extension = ".bin"

    return "mrpl_document" + extension


def looks_like_pdf(data: bytes) -> bool:
    """
    Real PDFs normally begin with %PDF-.
    """

    if not data:
        return False

    sample = data[:1024]

    return b"%PDF-" in sample


def looks_like_html(data: bytes) -> bool:
    if not data:
        return False

    sample = data[:2000].lower()

    html_signatures = [
        b"<!doctype html",
        b"<html",
        b"<head",
        b"<body",
        b"<script",
    ]

    return any(
        signature in sample
        for signature in html_signatures
    )


def download_document(url, existing_hashes):
    print()
    print("  DOCUMENT:")
    print(f"  {url}")

    for attempt in range(1, DOCUMENT_RETRIES + 1):

        print(
            f"  DOWNLOAD ATTEMPT "
            f"{attempt}/{DOCUMENT_RETRIES}"
        )

        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                headers={
                    "User-Agent": USER_AGENT,
                    "Referer": "https://mrpl.co.in/",
                    "Accept": (
                        "application/pdf,"
                        "application/msword,"
                        "application/vnd.openxmlformats-officedocument,"
                        "*/*"
                    ),
                },
            )

            status = response.status_code

            print(f"  HTTP status: {status}")
            print(
                f"  Final URL: "
                f"{response.url}"
            )

            if status != 200:
                print("  FAILED: HTTP error")

                time.sleep(2)

                continue

            data = response.content

            if not data:
                print("  FAILED: Empty response")

                time.sleep(2)

                continue

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            print(
                f"  Content-Type: "
                f"{content_type}"
            )

            # ------------------------------------------------
            # PDF VALIDATION
            # ------------------------------------------------

            if urlparse(url).path.lower().endswith(".pdf"):

                if not looks_like_pdf(data):
                    print(
                        "  REJECTED: NOT A REAL PDF "
                        "- missing %PDF- signature"
                    )

                    if looks_like_html(data):
                        print(
                            "  Server returned HTML "
                            "instead of PDF."
                        )

                    time.sleep(2)

                    continue

            # ------------------------------------------------
            # HTML VALIDATION
            # ------------------------------------------------

            if looks_like_html(data):
                print(
                    "  REJECTED: response appears "
                    "to be HTML"
                )

                time.sleep(2)

                continue

            # ------------------------------------------------
            # HASH CHECK
            # ------------------------------------------------

            digest = sha256_bytes(data)

            if digest in existing_hashes:
                print("  DUPLICATE HASH: skipped")

                return None

            # ------------------------------------------------
            # FILENAME
            # ------------------------------------------------

            filename = get_document_filename(
                url,
                content_type,
            )

            target = make_unique_path(
                DOC_DIR,
                filename,
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            with open(target, "wb") as f:
                f.write(data)

            existing_hashes.add(digest)

            print(
                f"  DOWNLOADED: "
                f"{target.name}"
            )

            print(
                f"  SIZE: "
                f"{len(data):,} bytes"
            )

            return target

        except requests.RequestException as exc:
            print(
                f"  REQUEST ERROR: {exc}"
            )

        except Exception as exc:
            print(
                f"  DOWNLOAD ERROR: {exc}"
            )

        time.sleep(2)

    print("  FAILED AFTER ALL RETRIES")

    return None


# ============================================================
# MANIFEST
# ============================================================

manifest_path = KB_DIR / "manifest.json"


def load_manifest():
    if not manifest_path.exists():
        return []

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


manifest = load_manifest()


def add_manifest(
    source_url,
    local_path,
    content_type,
    sha256,
    kind,
):
    record = {
        "source_url": source_url,
        "local_path": str(local_path),
        "content_type": content_type,
        "sha256": sha256,
        "kind": kind,
        "downloaded_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    manifest.append(record)

    try:
        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False,
            )

    except Exception as exc:
        print(
            f"  MANIFEST ERROR: {exc}"
        )


# ============================================================
# SCRAPLING FETCH
# ============================================================

def fetch_page(url):
    """
    Fetch one webpage with Scrapling.

    StealthyFetcher works correctly in your environment,
    as already confirmed by your successful test.
    """

    try:
        response = StealthyFetcher.fetch(
            url,
            headless=True,
            timeout=BROWSER_TIMEOUT,
        )

        return response

    except Exception as exc:
        print(
            f"  SCRAPLING ERROR: {exc}"
        )

        return None


# ============================================================
# CRAWLER
# ============================================================

def crawl():
    print("=" * 70)
    print("MRPL COMPLETE PUBLIC WEBSITE DOWNLOADER")
    print("SCRAPLING + DIRECT DOCUMENT DOWNLOAD")
    print("=" * 70)

    print()
    print("Starting URL:")
    print(START_URL)

    print()
    print("Knowledge Base:")
    print(KB_DIR)

    print()
    print("Rules:")
    print("  - MRPL domains only")
    print("  - English + Hindi pages allowed for discovery")
    print("  - Hindi pages are NOT discarded for crawling")
    print("  - Web page text is saved")
    print("  - Public documents are downloaded")
    print("  - PDF/DOC/DOCX/XLS/XLSX/CSV/PPT/PPTX/ZIP")
    print("  - Real PDF signature is verified")
    print("  - Duplicate hashes skipped")
    print("  - Existing files never overwritten")
    print("  - External websites rejected")
    print(f"  - Maximum pages: {MAX_PAGES}")
    print(f"  - Maximum depth: {MAX_DEPTH}")

    print()
    print("=" * 70)
    print("CHECKING EXISTING KNOWLEDGE BASE")
    print("=" * 70)

    existing_hashes = get_existing_hashes()

    existing_files = sum(
        1
        for root, _, files in os.walk(KB_DIR)
        for _ in files
    )

    print(
        f"Existing files: "
        f"{existing_files}"
    )

    print(
        f"Existing unique hashes: "
        f"{len(existing_hashes)}"
    )

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    start = normalize_url(START_URL)

    queue = deque()
    queue.append((start, 0))

    visited = set()

    pages_visited = 0
    documents_found = 0
    documents_downloaded = 0
    pages_saved = 0

    print()
    print("=" * 70)
    print("STARTING SCRAPLING")
    print("=" * 70)

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while queue and pages_visited < MAX_PAGES:

        current_url, depth = queue.popleft()

        current_url = normalize_url(current_url)

        if not current_url:
            continue

        if current_url in visited:
            continue

        if not is_mrpl_url(current_url):
            continue

        if should_skip_url(current_url):
            continue

        if depth > MAX_DEPTH:
            continue

        visited.add(current_url)

        pages_visited += 1

        print()
        print("=" * 70)
        print(
            f"VISITING "
            f"[{pages_visited} / {MAX_PAGES}]"
        )
        print("=" * 70)

        print(current_url)
        print(f"Depth: {depth}")

        # ----------------------------------------------------
        # DOCUMENT LINK
        # ----------------------------------------------------

        if is_document_url(current_url):

            documents_found += 1

            result = download_document(
                current_url,
                existing_hashes,
            )

            if result:
                documents_downloaded += 1

                try:
                    with open(
                        result,
                        "rb",
                    ) as f:
                        digest = sha256_bytes(
                            f.read()
                        )

                    add_manifest(
                        current_url,
                        result,
                        "document",
                        digest,
                        "document",
                    )

                except Exception:
                    pass

            time.sleep(CRAWL_DELAY)

            continue

        # ----------------------------------------------------
        # FETCH WEBPAGE
        # ----------------------------------------------------

        response = fetch_page(current_url)

        if response is None:
            print(
                "  PAGE FETCH FAILED"
            )

            continue

        try:
            status = response.status

        except Exception:
            status = None

        print(
            f"HTTP status: "
            f"{status}"
        )

        # ----------------------------------------------------
        # FINAL URL
        # ----------------------------------------------------

        try:
            final_url = normalize_url(
                str(response.url)
            )
        except Exception:
            final_url = current_url

        if final_url:
            print(
                f"Final URL: "
                f"{final_url}"
            )

        # ----------------------------------------------------
        # HINDI PAGE
        # ----------------------------------------------------

        is_hindi = (
            "/hi/" in final_url.lower()
            or "/hi" == urlparse(final_url).path.lower()
        )

        if is_hindi:
            print(
                "  HINDI PAGE DETECTED."
            )
            print(
                "  Keeping crawl active so "
                "linked documents are not lost."
            )

        # ----------------------------------------------------
        # TEXT EXTRACTION
        # ----------------------------------------------------

        text = extract_page_text(response)

        print(
            f"Text extracted: "
            f"{len(text):,} characters"
        )

        # Save text even if it is not one of our
        # preferred keywords. This prevents loss of
        # useful information.
        saved_page = save_page_text(
            final_url or current_url,
            text,
            existing_hashes,
        )

        if saved_page:

            pages_saved += 1

            try:
                digest = sha256_text(text)

                add_manifest(
                    final_url or current_url,
                    saved_page,
                    "text/plain",
                    digest,
                    "webpage",
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # EXTRACT LINKS
        # ----------------------------------------------------

        links = extract_links(
            response,
            final_url or current_url,
        )

        print(
            f"Links found: "
            f"{len(links)}"
        )

        # ----------------------------------------------------
        # ADD LINKS TO QUEUE
        # ----------------------------------------------------

        for link in links:

            if link in visited:
                continue

            if not is_mrpl_url(link):
                continue

            # Documents are always worth queueing.
            if is_document_url(link):

                documents_found += 1

                queue.append(
                    (
                        link,
                        depth + 1,
                    )
                )

                continue

            # Keep webpage crawl broad.
            queue.append(
                (
                    link,
                    depth + 1,
                )
            )

        print(
            f"Pages waiting in queue: "
            f"{len(queue)}"
        )

        time.sleep(CRAWL_DELAY)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MRPL SCRAPING COMPLETE")
    print("=" * 70)

    print(
        f"Pages visited: "
        f"{pages_visited}"
    )

    print(
        f"Web pages saved: "
        f"{pages_saved}"
    )

    print(
        f"Documents discovered: "
        f"{documents_found}"
    )

    print(
        f"New documents downloaded: "
        f"{documents_downloaded}"
    )

    print()
    print("KNOWLEDGE BASE:")
    print(KB_DIR)

    print()
    print("TEXT PAGES:")
    print(TEXT_DIR)

    print()
    print("DOCUMENTS:")
    print(DOC_DIR)

    print()
    print("MANIFEST:")
    print(manifest_path)

    print()
    print("Existing files were NOT deleted.")
    print("Existing files were NOT overwritten.")
    print("Duplicate files were skipped.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    crawl()