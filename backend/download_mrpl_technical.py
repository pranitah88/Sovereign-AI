"""
MRPL Focused Public Document Downloader
---------------------------------------

Uses Playwright so MRPL's browser/JS challenge can be handled.

Downloads useful public MRPL documents into:

D:\MRPL-Sovereign-AI\kb

Later, the contents of kb can be moved/merged into the RAG
knowledge_base if required.

Install:
    pip install playwright
    playwright install chromium

Run:
    python download_docs_playwright.py
"""

import os
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\MRPL-Sovereign-AI")

# 👇 EVERYTHING WILL BE DOWNLOADED HERE
OUTPUT_DIR = BASE_DIR / "kb"

START_URL = "https://mrpl.co.in/en/"

ALLOWED_DOMAIN = "mrpl.co.in"

# Safety limit — NOT 2000
MAX_PAGES = 100

REQUEST_DELAY = 1.0

NAV_TIMEOUT_MS = 45000

# Keep browser visible so you can see MRPL loading/challenge
HEADLESS = False

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0.0.0 Safari/537.36"
)

# Documents we actually want
DOC_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".csv",
    ".txt",
)

# ============================================================
# USEFUL MRPL AREAS
# ============================================================

USEFUL_KEYWORDS = [
    # Refining / manufacturing
    "manufacturing",
    "refining",
    "refinery",
    "processing",
    "process",
    "process-unit",
    "processing-unit",
    "hydrocracker",
    "distillation",
    "cdu",
    "vdu",
    "pfccu",
    "dcu",
    "dhdt",
    "hydrogen",
    "sulphur",
    "merox",
    "platforming",
    "propylene",
    "polypropylene",
    "bitumen",

    # Engineering / technical
    "engineering",
    "technology",
    "technical",
    "maintenance",
    "inspection",
    "equipment",
    "pipeline",
    "capacity",
    "production",

    # Safety
    "safety",
    "hse",
    "health",
    "fire",
    "emergency",
    "work-permit",
    "permit",

    # Environment
    "environment",
    "environmental",
    "emission",
    "effluent",
    "waste",
    "hazardous",

    # Projects
    "project",
    "projects",
    "expansion",
    "modernization",

    # Reports
    "annual-report",
    "annualreport",
    "report",
]


# ============================================================
# URL HELPERS
# ============================================================

def normalize_url(url):
    """
    Remove fragments and trailing spaces.
    """
    return url.split("#")[0].strip()


def is_same_domain(url):
    try:
        hostname = urlparse(url).netloc.lower()

        return (
            hostname == ALLOWED_DOMAIN
            or hostname.endswith("." + ALLOWED_DOMAIN)
        )

    except Exception:
        return False


def is_hindi_url(url):
    """
    Prevent crawler from following Hindi pages.
    """
    path = urlparse(url).path.lower()

    return (
        "/hi/" in path
        or path.startswith("/hi")
    )


def is_document(url):
    path = urlparse(url).path.lower()

    return path.endswith(DOC_EXTENSIONS)


def looks_useful(url):
    """
    Check whether a URL appears relevant to our
    MRPL technical/RAG knowledge base.
    """

    value = url.lower()

    return any(
        keyword in value
        for keyword in USEFUL_KEYWORDS
    )


def safe_filename(url):
    """
    Create a safe filename from URL.
    """

    parsed = urlparse(url)

    filename = os.path.basename(parsed.path)

    if not filename:
        filename = "mrpl_document"

    # Remove query-like unsafe characters
    filename = filename.replace(":", "_")
    filename = filename.replace("/", "_")
    filename = filename.replace("\\", "_")

    # Short hash prevents same-name collisions
    url_hash = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:10]

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    return f"{stem}_{url_hash}{suffix}"


# ============================================================
# HASHING
# ============================================================

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def get_existing_hashes():
    """
    Scan kb folder for files already downloaded.
    """

    hashes = set()

    if not OUTPUT_DIR.exists():
        return hashes

    print("\nChecking existing files...")

    for file in OUTPUT_DIR.rglob("*"):

        if not file.is_file():
            continue

        try:

            with open(file, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()

            hashes.add(digest)

        except Exception:
            pass

    print(
        f"Existing file hashes found: {len(hashes)}"
    )

    return hashes


# ============================================================
# DOWNLOAD DOCUMENT
# ============================================================

def download_document(context, url, existing_hashes):
    """
    Download a document using Playwright's browser context.

    This is important because it reuses the browser session
    and cookies used to pass MRPL's challenge.
    """

    filename = safe_filename(url)

    output_file = OUTPUT_DIR / filename

    # Already exists by filename
    if output_file.exists():

        print(
            f"  EXISTS: {output_file.name}"
        )

        return False

    try:

        print()
        print("  DOCUMENT:")
        print(f"  {url}")

        response = context.request.get(
            url,
            timeout=NAV_TIMEOUT_MS
        )

        print(
            f"  HTTP status: {response.status}"
        )

        if response.status != 200:

            print(
                "  DOWNLOAD FAILED"
            )

            return False

        data = response.body()

        if not data:

            print(
                "  Empty response."
            )

            return False

        # Duplicate by actual file contents
        digest = sha256_bytes(data)

        if digest in existing_hashes:

            print(
                "  SKIPPED DUPLICATE HASH"
            )

            return False

        # Save file
        with open(output_file, "wb") as f:
            f.write(data)

        existing_hashes.add(digest)

        size_kb = len(data) / 1024

        print(
            f"  SAVED: {output_file}"
        )

        print(
            f"  SIZE: {size_kb:.1f} KB"
        )

        return True

    except Exception as e:

        print(
            f"  DOWNLOAD ERROR: {e}"
        )

        return False


# ============================================================
# GET LINKS FROM PAGE
# ============================================================

def get_page_links(page, current_url):

    links = set()

    try:

        hrefs = page.eval_on_selector_all(
            "a[href]",
            """
            elements =>
            elements.map(
                element => element.getAttribute("href")
            )
            """
        )

    except Exception:

        return links

    for href in hrefs:

        if not href:
            continue

        full_url = normalize_url(
            urljoin(current_url, href)
        )

        if not full_url.startswith(
            ("http://", "https://")
        ):
            continue

        if not is_same_domain(full_url):
            continue

        if is_hindi_url(full_url):
            continue

        links.add(full_url)

    return links


# ============================================================
# CRAWLER
# ============================================================

def crawl():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("MRPL FOCUSED PUBLIC DOCUMENT DOWNLOADER")
    print("=" * 70)

    print()
    print("Download folder:")
    print(OUTPUT_DIR)

    print()
    print("Maximum pages:")
    print(MAX_PAGES)

    print()
    print("Starting browser...")

    visited_pages = set()

    discovered_documents = set()

    queue = [
        START_URL,

        # Important English pages
        "https://mrpl.co.in/en/Content/Manufacturing_Units",
        "https://mrpl.co.in/en/Content/Refining",
        "https://mrpl.co.in/en/MenuContent/MRPL_CUTTING_EDGE_TECHNOLOGY_IN_REFINING",
        "https://mrpl.co.in/en/Content/Health%20and%20Safety",
        "https://mrpl.co.in/en/Content/Environment%20Care",
        "https://mrpl.co.in/en/EnvironmentRelatedData",
        "https://mrpl.co.in/en/Content/Projects",
        "https://mrpl.co.in/en/AnnualReport",
    ]

    existing_hashes = get_existing_hashes()

    downloaded_count = 0

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=HEADLESS
        )

        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={
                "width": 1366,
                "height": 768
            },
            locale="en-IN",
            timezone_id="Asia/Kolkata",
        )

        page = context.new_page()

        # ----------------------------------------------------
        # Crawl
        # ----------------------------------------------------

        while queue and len(visited_pages) < MAX_PAGES:

            url = queue.pop(0)

            url = normalize_url(url)

            if url in visited_pages:
                continue

            if is_hindi_url(url):
                continue

            visited_pages.add(url)

            print()
            print("=" * 70)
            print(
                f"VISITING [{len(visited_pages)} / {MAX_PAGES}]"
            )
            print(url)
            print("=" * 70)

            try:

                response = page.goto(
                    url,
                    timeout=NAV_TIMEOUT_MS,
                    wait_until="domcontentloaded"
                )

                if response is None:

                    print(
                        "No response."
                    )

                    continue

                print(
                    f"HTTP status: {response.status}"
                )

                # --------------------------------------------
                # Handle MRPL challenge
                # --------------------------------------------

                if response.status == 503:

                    print(
                        "  MRPL challenge detected."
                    )

                    print(
                        "  Waiting for browser challenge..."
                    )

                    try:

                        page.wait_for_timeout(
                            7000
                        )

                    except Exception:
                        pass

                    print(
                        "  Current URL:",
                        page.url
                    )

                # --------------------------------------------
                # Check final URL
                # --------------------------------------------

                final_url = normalize_url(
                    page.url
                )

                print(
                    "Final URL:",
                    final_url
                )

                if is_hindi_url(final_url):

                    print(
                        "  SKIPPED HINDI PAGE"
                    )

                    continue

                # --------------------------------------------
                # Wait for page content
                # --------------------------------------------

                page.wait_for_timeout(
                    1500
                )

                # --------------------------------------------
                # Get content type
                # --------------------------------------------

                try:

                    content_type = (
                        response.headers.get(
                            "content-type",
                            ""
                        )
                        .lower()
                    )

                except Exception:

                    content_type = ""

                # --------------------------------------------
                # If direct document
                # --------------------------------------------

                if is_document(final_url):

                    if final_url not in discovered_documents:

                        discovered_documents.add(
                            final_url
                        )

                        if download_document(
                            context,
                            final_url,
                            existing_hashes
                        ):

                            downloaded_count += 1

                    continue

                # --------------------------------------------
                # HTML page
                # --------------------------------------------

                if (
                    "text/html"
                    not in content_type
                ):

                    print(
                        "  Not an HTML page."
                    )

                    continue

                # --------------------------------------------
                # Extract links
                # --------------------------------------------

                links = get_page_links(
                    page,
                    final_url
                )

                print(
                    f"Links found: {len(links)}"
                )

                # --------------------------------------------
                # Process links
                # --------------------------------------------

                for link in links:

                    # Document
                    if is_document(link):

                        if link in discovered_documents:
                            continue

                        discovered_documents.add(
                            link
                        )

                        # Only download useful documents
                        if looks_useful(link):

                            if download_document(
                                context,
                                link,
                                existing_hashes
                            ):

                                downloaded_count += 1

                            time.sleep(
                                REQUEST_DELAY
                            )

                    # HTML page
                    else:

                        if link in visited_pages:
                            continue

                        # Only queue useful-looking pages
                        if looks_useful(link):

                            queue.append(link)

                print(
                    f"Pages waiting in queue: {len(queue)}"
                )

                time.sleep(
                    REQUEST_DELAY
                )

            except PWTimeout:

                print(
                    "  PAGE TIMEOUT"
                )

            except Exception as e:

                print(
                    "  PAGE ERROR:",
                    e
                )

        browser.close()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("MRPL DOWNLOAD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Pages visited: {len(visited_pages)}"
    )

    print(
        f"Documents discovered: {len(discovered_documents)}"
    )

    print(
        f"New documents downloaded: {downloaded_count}"
    )

    print()
    print("FILES ARE SAVED DIRECTLY HERE:")
    print(OUTPUT_DIR)

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    crawl()