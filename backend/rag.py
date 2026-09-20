import os

# Disable Paddle's oneDNN / MKL-DNN path before importing PaddleOCR.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import hashlib
import io
import json
import re
from pathlib import Path

import chromadb
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

from models import ask_gemma


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("D:/MRPL-Sovereign-AI")

KB_DIR = BASE_DIR / "merged_knowledge_base"
CHROMA_DIR = BASE_DIR / "backend" / "chroma_db"

COLLECTION_NAME = "mrpl_knowledge_base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieve a larger pool before reranking.
INITIAL_TOP_K = 30

# Number of final sources sent to Gemma.
FINAL_TOP_K = 8

# Maximum acceptable Chroma distance.
MAX_DISTANCE_THRESHOLD = 0.90

# Pages with less native text than this are treated as scanned/poorly extracted.
OCR_MIN_NATIVE_CHARS = 80

# OCR rendering resolution.
OCR_DPI = 100


# ============================================================
# OPTIONAL PADDLEOCR
# ============================================================

PADDLE_AVAILABLE = False
ocr = None

try:
    from paddleocr import PaddleOCR

    print("Loading PaddleOCR...")

    # GPU configuration validated on the current setup.
    ocr = PaddleOCR(
        device="gpu",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    PADDLE_AVAILABLE = True

    print("PaddleOCR loaded successfully.")

except Exception as exc:
    print("WARNING: PaddleOCR could not be loaded.")
    print(f"Reason: {exc}")
    print("Scanned PDFs will not receive OCR.")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device="cpu",
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text).replace("\x00", " ")

    # Normalize spaces while preserving paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_dictionary_text(data):
    """
    Best-effort extraction of OCR text from PaddleOCR result dictionaries.
    """

    found = []

    if not isinstance(data, dict):
        return found

    for key in (
        "rec_texts",
        "texts",
        "text",
        "ocr_text",
    ):
        value = data.get(key)

        if isinstance(value, str):
            if value.strip():
                found.append(value.strip())

        elif isinstance(value, (list, tuple)):
            for item in value:

                if isinstance(item, str):
                    if item.strip():
                        found.append(item.strip())

                elif isinstance(item, dict):
                    found.extend(
                        extract_dictionary_text(item)
                    )

    # Recursively inspect nested structures.
    for value in data.values():

        if isinstance(value, dict):
            found.extend(
                extract_dictionary_text(value)
            )

        elif isinstance(value, list):
            for item in value:

                if isinstance(item, dict):
                    found.extend(
                        extract_dictionary_text(item)
                    )

    return found


def extract_paddle_text(result):
    """
    Handle common PaddleOCR 3.x result representations.
    """

    texts = []

    if result is None:
        return ""

    # List/tuple result
    if isinstance(result, (list, tuple)):

        for item in result:

            text = extract_paddle_text(item)

            if text:
                texts.append(text)

        return clean_text(
            "\n".join(texts)
        )

    # Dictionary result
    if isinstance(result, dict):

        texts.extend(
            extract_dictionary_text(result)
        )

        return clean_text(
            "\n".join(texts)
        )

    # PaddleOCR result object may expose .json
    try:

        data = getattr(
            result,
            "json",
            None,
        )

        if callable(data):
            data = data()

        if isinstance(data, str):

            try:
                data = json.loads(data)
            except Exception:
                pass

        if isinstance(data, (dict, list)):

            text = extract_paddle_text(data)

            if text:
                texts.append(text)

    except Exception:
        pass

    # Fallback to .rec_texts
    if not texts:

        try:

            rec_texts = getattr(
                result,
                "rec_texts",
                None,
            )

            if isinstance(
                rec_texts,
                (list, tuple),
            ):

                texts.extend(
                    str(item).strip()
                    for item in rec_texts
                    if str(item).strip()
                )

        except Exception:
            pass

    return clean_text(
        "\n".join(texts)
    )


# ============================================================
# OCR
# ============================================================

def run_paddle_ocr_on_page(page):
    """
    Render one PDF page and run PaddleOCR locally.
    """

    if not PADDLE_AVAILABLE:
        return ""

    try:

        pix = page.get_pixmap(
            dpi=OCR_DPI,
            alpha=False,
        )

        from PIL import Image

        image = Image.open(
            io.BytesIO(
                pix.tobytes("png")
            )
        ).convert("RGB")

        image_np = np.asarray(
            image,
            dtype=np.uint8,
        )

        result = ocr.predict(
            image_np
        )

        return extract_paddle_text(
            result
        )

    except Exception as exc:

        print(
            f"        OCR failed: {exc}"
        )

        return ""


# ============================================================
# FILE READING
# ============================================================

def read_text_file(file_path):

    try:

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception as exc:

        print(
            f"Could not read {file_path}: {exc}"
        )

        return []

    text = clean_text(text)

    if not text:
        return []

    return [
        {
            "page": 1,
            "text": text,
            "ocr": False,
        }
    ]


def read_json_file(file_path):

    try:

        raw = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        data = json.loads(raw)

        text = clean_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        print(
            f"Could not read JSON {file_path}: {exc}"
        )

        return []

    if not text:
        return []

    return [
        {
            "page": 1,
            "text": text,
            "ocr": False,
        }
    ]


def read_pdf_file(file_path):

    pages = []

    try:

        document = pymupdf.open(
            file_path
        )

    except Exception as exc:

        print(
            f"Could not open PDF {file_path}: {exc}"
        )

        return []

    try:

        total_pages = document.page_count

        print(
            f"        PDF pages: {total_pages}"
        )

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            native_text = clean_text(
                page.get_text("text")
            )

            # ------------------------------------------------
            # Native text
            # ------------------------------------------------

            if len(native_text) >= OCR_MIN_NATIVE_CHARS:

                print(
                    f"        Page {page_number}: "
                    f"native text "
                    f"({len(native_text.split())} words)"
                )

                pages.append(
                    {
                        "page": page_number,
                        "text": native_text,
                        "ocr": False,
                    }
                )

                continue

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            print(
                f"        Page {page_number}: OCR"
            )

            ocr_text = run_paddle_ocr_on_page(
                page
            )

            if ocr_text:

                print(
                    f"        OCR result: "
                    f"{len(ocr_text.split())} words"
                )

                pages.append(
                    {
                        "page": page_number,
                        "text": ocr_text,
                        "ocr": True,
                    }
                )

            elif native_text:

                print(
                    "        OCR returned no text; "
                    "keeping native text"
                )

                pages.append(
                    {
                        "page": page_number,
                        "text": native_text,
                        "ocr": False,
                    }
                )

            else:

                print(
                    f"        Page {page_number}: NO TEXT"
                )

    finally:

        document.close()

    return pages


def read_file(file_path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf_file(file_path)

    if suffix in {
        ".txt",
        ".md",
        ".log",
    }:
        return read_text_file(file_path)

    if suffix == ".json":
        return read_json_file(file_path)

    return []


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    text = re.sub(
        r"\s+",
        " ",
        clean_text(text),
    ).strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# METADATA
# ============================================================

def get_category(file_path):

    try:

        relative_path = file_path.relative_to(
            KB_DIR
        )

        parts = relative_path.parts

        if len(parts) > 1:
            return "/".join(parts[:-1])

        return "root"

    except Exception:

        return "unknown"


def detect_year(source):

    match = re.search(
        r"(20\d{2})",
        source,
    )

    return (
        match.group(1)
        if match
        else "Unknown"
    )


def detect_document_type(
    source,
    category="",
):

    value = f"{source} {category}".lower()

    if "annual report" in value:
        return "annual_report"

    if (
        "consent" in value
        or "cfo" in value
    ):
        return "environment"

    if "inspection" in value:
        return "inspection"

    if (
        "safety" in value
        or "hse" in value
    ):
        return "safety"

    if "financial result" in value:
        return "financial_result"

    if (
        "financial" in value
        or "finance" in value
    ):
        return "financial"

    if (
        "environment" in value
        or "pollution" in value
    ):
        return "environment"

    if (
        "project" in value
        or "expansion" in value
    ):
        return "project"

    if (
        "manufacturing" in value
        or "refining" in value
    ):
        return "manufacturing_refining"

    if "csr" in value:
        return "csr"

    if "statutory" in value:
        return "statutory"

    return "general"


def detect_status(
    source,
    category="",
):

    """
    Best-effort classification.

    operational:
        Existing/current information.

    proposed:
        Planned/proposed/project information.

    unspecified:
        No strong signal from filename/category.
    """

    value = f"{source} {category}".lower()

    proposed_terms = [
        "proposed",
        "phase-iii",
        "phase iii",
        "expansion project",
        "consent for establishment",
        "cfe",
    ]

    operational_terms = [
        "annual report",
        "official website",
        "brochure",
        "current updates",
        "refining",
        "manufacturing",
    ]

    # Strong proposal indicators first.
    if any(
        term in value
        for term in proposed_terms
    ):
        return "proposed"

    if any(
        term in value
        for term in operational_terms
    ):
        return "operational"

    return "unspecified"


def create_metadata(
    file_path,
    page_number,
    ocr_used,
):

    source = file_path.name

    category = get_category(
        file_path
    )

    return {
        "source": source,

        "path": str(file_path),

        "category": category,

        "document_type": detect_document_type(
            source,
            category,
        ),

        "status": detect_status(
            source,
            category,
        ),

        "year": detect_year(
            source
        ),

        "page": int(page_number),

        "ocr": bool(
            ocr_used
        ),
    }


def create_chunk_id(
    source,
    page,
    chunk_number,
    text,
):

    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_number}|"
        f"{text}"
    )

    return hashlib.md5(
        raw.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():

    print()
    print("=" * 70)
    print(
        "BUILDING / REBUILDING MRPL KNOWLEDGE BASE"
    )
    print("=" * 70)

    print(
        f"Knowledge base: {KB_DIR}"
    )

    if not KB_DIR.exists():

        print(
            "ERROR: Knowledge base folder "
            "does not exist."
        )

        return

    supported_extensions = {
        ".pdf",
        ".txt",
        ".md",
        ".json",
        ".log",
    }

    files = sorted(
        path
        for path in KB_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in supported_extensions
        )
    )

    print(
        f"Files found: {len(files)}"
    )

    if not files:

        print(
            "No supported files found."
        )

        return

    global collection

    # Option 1 intentionally deletes the old collection.
    try:

        chroma_client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Old ChromaDB collection deleted."
        )

    except Exception:
        pass

    collection = (
        chroma_client
        .get_or_create_collection(
            name=COLLECTION_NAME
        )
    )

    all_ids = []
    all_documents = []
    all_metadatas = []

    total_pages = 0
    total_chunks = 0
    total_ocr_pages = 0

    # ========================================================
    # PROCESS FILES
    # ========================================================

    for file_number, file_path in enumerate(
        files,
        start=1,
    ):

        print()
        print(
            f"[{file_number}/{len(files)}] "
            f"{file_path.name}"
        )

        pages = read_file(
            file_path
        )

        file_chunk_count = 0

        for page_data in pages:

            page_number = page_data[
                "page"
            ]

            page_text = clean_text(
                page_data["text"]
            )

            ocr_used = bool(
                page_data.get(
                    "ocr",
                    False,
                )
            )

            if not page_text:
                continue

            total_pages += 1

            total_ocr_pages += int(
                ocr_used
            )

            chunks = create_chunks(
                page_text
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1,
            ):

                chunk_id = create_chunk_id(
                    file_path.name,
                    page_number,
                    chunk_number,
                    chunk,
                )

                all_ids.append(
                    chunk_id
                )

                all_documents.append(
                    chunk
                )

                all_metadatas.append(
                    create_metadata(
                        file_path,
                        page_number,
                        ocr_used,
                    )
                    | {
                        "chunk": chunk_number
                    }
                )

                file_chunk_count += 1
                total_chunks += 1

        print(
            f"        Chunks created: "
            f"{file_chunk_count}"
        )

    if not all_documents:

        print(
            "ERROR: No chunks were created."
        )

        return

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING EMBEDDINGS")
    print("=" * 70)

    batch_size = 32

    for start in range(
        0,
        len(all_documents),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(all_documents),
        )

        embeddings = embedding_model.encode(
            all_documents[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        collection.add(
            ids=all_ids[start:end],

            documents=all_documents[
                start:end
            ],

            metadatas=all_metadatas[
                start:end
            ],

            embeddings=np.asarray(
                embeddings
            ).tolist(),
        )

        print(
            f"Indexed "
            f"{end}/{len(all_documents)} "
            f"chunks"
        )

    print()
    print("=" * 70)
    print("KNOWLEDGE BASE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Files processed : {len(files)}"
    )

    print(
        f"Pages indexed   : {total_pages}"
    )

    print(
        f"OCR pages       : {total_ocr_pages}"
    )

    print(
        f"Chunks indexed  : {total_chunks}"
    )

    print(
        f"Database        : {CHROMA_DIR}"
    )


# ============================================================
# QUERY ANALYSIS
# ============================================================

def analyse_query(query):

    query_lower = query.lower()

    analysis = {
        "annual_report": False,
        "financial": False,
        "safety": False,
        "environment": False,
        "project": False,
        "inspection": False,
        "manufacturing": False,
        "consent": False,
        "ocr": False,
        "year": None,
        "report_number": None,
    }

    # --------------------------------------------------------
    # Annual report
    # --------------------------------------------------------

    if (
        "annual report" in query_lower
        or "annual reports" in query_lower
    ):

        analysis[
            "annual_report"
        ] = True

    # --------------------------------------------------------
    # Financial
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "financial",
            "revenue",
            "profit",
            "loss",
            "turnover",
            "ebitda",
            "cash flow",
            "balance sheet",
        ]
    ):

        analysis[
            "financial"
        ] = True

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "safety",
            "permit",
            "hse",
            "fire",
            "hazard",
            "ppe",
            "confined space",
            "hot work",
        ]
    ):

        analysis[
            "safety"
        ] = True

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "environment",
            "environmental",
            "emission",
            "waste",
            "hazardous waste",
            "effluent",
            "pollution",
            "water",
            "air",
            "consent",
        ]
    ):

        analysis[
            "environment"
        ] = True

    # --------------------------------------------------------
    # Consent
    # --------------------------------------------------------

    if any(
        phrase in query_lower
        for phrase in [
            "consent",
            "consent for operation",
            "cfo",
            "kspcb",
            "pollution control board",
        ]
    ):

        analysis[
            "consent"
        ] = True

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "project",
            "projects",
            "pipeline",
            "upgrade",
            "expansion",
        ]
    ):

        analysis[
            "project"
        ] = True

    # --------------------------------------------------------
    # Inspection
    # --------------------------------------------------------

    if any(
        word in query_lower
        for word in [
            "inspection",
            "inspection report",
            "equipment condition",
            "finding",
            "defect",
        ]
    ):

        analysis[
            "inspection"
        ] = True

    # --------------------------------------------------------
    # Manufacturing / refining
    # --------------------------------------------------------

    technical_terms = [
        "cdu",
        "vdu",
        "dcu",
        "pfccu",
        "dhdt",
        "hydrocracker",
        "hydrogen",
        "sru",
        "merox",
        "platforming",
        "ccr",
        "visbreaker",
        "bitumen",
        "propylene",
        "polypropylene",
        "refinery",
        "refining",
        "processing unit",
        "processing units",
        "manufacturing unit",
        "manufacturing units",
    ]

    if any(
        term in query_lower
        for term in technical_terms
    ):

        analysis[
            "manufacturing"
        ] = True

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    if (
        "ocr" in query_lower
        or "scanned" in query_lower
    ):

        analysis[
            "ocr"
        ] = True

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    year_match = re.search(
        r"(20\d{2})",
        query,
    )

    if year_match:

        analysis[
            "year"
        ] = year_match.group(1)

    # --------------------------------------------------------
    # Annual report number
    # --------------------------------------------------------

    report_match = re.search(
        r"(\d+)(?:st|nd|rd|th)?\s+annual report",
        query_lower,
    )

    if report_match:

        analysis[
            "report_number"
        ] = report_match.group(1)

    return analysis


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    query,
    text,
):

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower(),
        )
    )

    text_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )

    if not query_words:
        return 0.0

    return (
        len(
            query_words
            & text_words
        )
        / len(query_words)
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(query):

    if collection.count() == 0:

        print()
        print(
            "WARNING: ChromaDB collection "
            "is empty."
        )

        print(
            "Please rebuild the knowledge "
            "base first."
        )

        return []

    analysis = analyse_query(
        query
    )

    # --------------------------------------------------------
    # Embed user query
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        .tolist()
    )

    # --------------------------------------------------------
    # Query ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],

        n_results=min(
            INITIAL_TOP_K,
            collection.count(),
        ),

        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    # ========================================================
    # TEMPORARY DEBUG
    # ========================================================

    print()
    print(
        "[DEBUG] Raw ChromaDB retrieval"
    )

    print(
        f"[DEBUG] Documents returned: "
        f"{len(documents)}"
    )

    for i, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        print(
            f"[DEBUG] {i}. "
            f"Distance={float(distance):.4f} | "
            f"Source={metadata.get('source', 'Unknown')}"
        )

    # ========================================================
    # RERANK
    # ========================================================

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        source = metadata.get(
            "source",
            "",
        )

        category = metadata.get(
            "category",
            "",
        )

        document_type = metadata.get(
            "document_type",
            "general",
        )

        year = metadata.get(
            "year",
            "Unknown",
        )

        source_lower = source.lower()
        category_lower = category.lower()
        document_lower = document.lower()

        combined_lower = (
            f"{source} "
            f"{category} "
            f"{document}"
        ).lower()

        # ----------------------------------------------------
        # Base score
        # ----------------------------------------------------

        semantic_score = max(
            0.0,
            1.0 - float(distance),
        )

        score = (
            semantic_score * 5.0
        )

        score += (
            keyword_score(
                query,
                document,
            )
            * 5.0
        )

        # ----------------------------------------------------
        # Prefer MRPL
        # ----------------------------------------------------

        if "ompl" in source_lower:

            score -= 8

        elif "mrpl" in source_lower:

            score += 3

        # ----------------------------------------------------
        # Annual reports
        # ----------------------------------------------------

        if analysis[
            "annual_report"
        ]:

            if (
                document_type
                == "annual_report"
            ):

                score += 7

            else:

                score -= 1

        # ----------------------------------------------------
        # Manufacturing / refining
        # ----------------------------------------------------

        if analysis[
            "manufacturing"
        ]:

            if (
                document_type
                == "manufacturing_refining"
            ):

                score += 8

            refinery_terms = [
                "cdu",
                "vdu",
                "dcu",
                "pfccu",
                "dhdt",
                "hydrocracker",
                "sru",
                "platforming",
                "merox",
                "visbreaker",
                "refinery",
                "refining",
                "processing unit",
            ]

            score += sum(
                1.5
                for term in refinery_terms
                if term in document_lower
            )

        # ----------------------------------------------------
        # Financial
        # ----------------------------------------------------

        if analysis[
            "financial"
        ]:

            if (
                document_type
                in {
                    "financial",
                    "financial_result",
                    "annual_report",
                }
                or "finance"
                in category_lower
            ):

                score += 7

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if analysis[
            "safety"
        ]:

            if (
                document_type == "safety"
                or "safety"
                in category_lower
            ):

                score += 10

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        if analysis[
            "environment"
        ]:

            if (
                document_type
                == "environment"
                or "environment"
                in category_lower
                or "compliance"
                in category_lower
            ):

                score += 12

            environment_terms = [
                "environment",
                "environmental",
                "emission",
                "pollution",
                "waste",
                "effluent",
                "consent",
                "kspcb",
                "water",
                "air",
            ]

            score += sum(
                1.5
                for term in environment_terms
                if term in combined_lower
            )

        # ----------------------------------------------------
        # Consent
        # ----------------------------------------------------

        if analysis[
            "consent"
        ]:

            if (
                "environment"
                in category_lower
                or "compliance"
                in category_lower
                or document_type
                == "environment"
            ):

                score += 14

            consent_terms = [
                "consent",
                "cfo",
                "kspcb",
                "pollution control board",
                "consent for operation",
            ]

            score += sum(
                3.0
                for term in consent_terms
                if term in combined_lower
            )

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        if analysis[
            "project"
        ]:

            if (
                document_type
                == "project"
                or "project"
                in category_lower
            ):

                score += 8

        # ----------------------------------------------------
        # Inspection
        # ----------------------------------------------------

        if analysis[
            "inspection"
        ]:

            if (
                document_type
                == "inspection"
                or "inspection"
                in combined_lower
            ):

                score += 10

        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------

        if analysis[
            "year"
        ]:

            if (
                year
                == analysis[
                    "year"
                ]
            ):

                score += 6

        # ----------------------------------------------------
        # Annual report number
        # ----------------------------------------------------

        if analysis[
            "report_number"
        ]:

            if (
                analysis[
                    "report_number"
                ]
                in source_lower
            ):

                score += 7

        # ----------------------------------------------------
        # OCR preference
        # ----------------------------------------------------

        if (
            analysis["ocr"]
            and metadata.get(
                "ocr",
                False,
            )
        ):

            score += 5

        candidates.append(
            {
                "score": score,
                "document": document,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    candidates.sort(
        key=lambda item: item[
            "score"
        ],
        reverse=True,
    )

    return candidates


# ============================================================
# SELECT BEST CHUNKS
# ============================================================

def select_best_chunks(
    candidates
):

    selected = []
    seen = set()

    for candidate in candidates:

        # Reject weak semantic matches.
        if (
            candidate["distance"]
            > MAX_DISTANCE_THRESHOLD
        ):
            continue

        metadata = candidate[
            "metadata"
        ]

        # One chunk per source/page.
        key = (
            metadata.get(
                "source",
                "",
            ),
            metadata.get(
                "page",
                "",
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        selected.append(
            candidate
        )

        if len(selected) >= FINAL_TOP_K:
            break

    return selected


# ============================================================
# DISPLAY RETRIEVED SOURCES
# ============================================================

def display_retrieved_documents(
    candidates
):

    print()
    print("=" * 80)
    print(
        "TOP RETRIEVED SOURCES"
    )
    print("=" * 80)

    if not candidates:

        print(
            "No documents retrieved."
        )

        return

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        metadata = candidate[
            "metadata"
        ]

        print()

        print(
            f"#{index} "
            f"Score: "
            f"{candidate['score']:.2f}"
        )

        print(
            f"Source: "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Category: "
            f"{metadata.get('category', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 'Unknown')}"
        )

        print(
            f"Type: "
            f"{metadata.get('document_type', 'Unknown')}"
        )

        print(
            f"Status: "
            f"{metadata.get('status', 'unspecified')}"
        )

        print(
            f"Year: "
            f"{metadata.get('year', 'Unknown')}"
        )

        print(
            f"OCR: "
            f"{metadata.get('ocr', False)}"
        )

        print(
            f"Distance: "
            f"{candidate['distance']:.4f}"
        )

        print()

        print(
            candidate["document"][:1400]
        )

        print(
            "-" * 80
        )


# ============================================================
# BUILD GEMMA CONTEXT
# ============================================================

def build_context(
    candidates
):

    parts = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        metadata = candidate[
            "metadata"
        ]

        parts.append(
            f"""
SOURCE {index}

Document:
{metadata.get('source', 'Unknown')}

Category:
{metadata.get('category', 'Unknown')}

Document type:
{metadata.get('document_type', 'Unknown')}

Status:
{metadata.get('status', 'unspecified')}

Year:
{metadata.get('year', 'Unknown')}

Page:
{metadata.get('page', 'Unknown')}

OCR used:
{metadata.get('ocr', False)}

Content:
{candidate['document']}
"""
        )

    return "\n".join(parts)


# ============================================================
# ASK RAG
# ============================================================

def ask_rag(
    question
):

    print()
    print(
        "Searching local knowledge base..."
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        question
    )

    # --------------------------------------------------------
    # DEBUG: show top 5 raw candidates
    # --------------------------------------------------------

    print()
    print(
        "[DEBUG] Top 5 raw candidate distances:"
    )

    for candidate in candidates[:5]:

        print(
            f"  "
            f"{candidate['distance']:.4f}  "
            f"{candidate['metadata'].get('source', 'Unknown')}"
        )

    # --------------------------------------------------------
    # Select
    # --------------------------------------------------------

    selected = select_best_chunks(
        candidates
    )

    # --------------------------------------------------------
    # No suitable sources
    # --------------------------------------------------------

    if not selected:

        print()

        print(
            "I could not find relevant "
            "information in the local "
            "knowledge base."
        )

        print()

        print(
            "[DEBUG] No candidate passed "
            f"the distance threshold of "
            f"{MAX_DISTANCE_THRESHOLD}."
        )

        return

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_retrieved_documents(
        selected
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        selected
    )

    # ========================================================
    # GEMMA PROMPT
    # ========================================================

    prompt = f"""
You are the local AI assistant inside a
Sovereign On-Premise Industrial AI Workbench.

Answer the user's question using ONLY the
LOCAL KNOWLEDGE BASE provided below.

============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY the retrieved local sources below.

2. NEVER use outside knowledge, web knowledge,
   model memory, or assumptions.

3. NEVER invent facts, numbers, dates, capacities,
   technical details, names, or terminology.

4. Preserve technical names, abbreviations, units,
   capacities, dates, and terminology exactly as
   supported by the sources.

5. NEVER expand an abbreviation unless a retrieved
   source explicitly gives that expansion.

6. Pay close attention to document status:

   operational = existing/current information
   proposed    = planned/proposed/project information
   unspecified  = status cannot be established

7. NEVER merge operational and proposed information
   into one undifferentiated list.

8. If both operational and proposed information are
   relevant, create separate clearly labelled sections.

9. Do NOT treat a project proposal, environmental
   clearance document, CFE document, or planning
   document as proof that a unit or activity is
   currently operating.

10. Prefer the source that directly answers the question.

11. Prefer official MRPL sources when they directly
    answer the question.

12. When an older document and a newer document
    contain different information, mention the
    relevant year and explain the difference.

13. For EVERY factual statement, provide an inline
    citation using exactly this format:

    [Source: filename, Page X]

14. Do not create citations for information that is
    not actually present in the retrieved source.

15. If OCR text appears corrupted or unclear, do not
    silently reconstruct it.

16. If the retrieved evidence only partially answers
    the question, explicitly explain what is supported
    and what is not supported.

17. Never claim that a document is confidential,
    internal, restricted, classified, or secret unless
    the retrieved source explicitly states this.

18. Do not combine information from different years
    as though it belongs to one period.

19. For questions asking about "current" or "existing"
    information, prioritize sources identified as
    operational.

20. For questions asking about a project, expansion,
    proposal, or Phase-III, proposed documents may
    receive priority when they directly answer the
    question.

============================================================
TECHNICAL TERMINOLOGY
============================================================

Preserve technical abbreviations exactly.

For example, if a source explicitly states:

DCU = Delayed Coker Unit

then write:

Delayed Coker Unit

Do NOT reinterpret DCU as another expansion.

Never silently redefine a technical abbreviation.

============================================================
SOURCE SELECTION
============================================================

For a question such as:

"What are the major process units in MRPL's refinery?"

prefer official MRPL documents describing the
refinery/process configuration.

Do NOT answer primarily from a Phase-III proposal
unless the question asks about Phase-III or proposed units.

============================================================
ANSWER FORMAT
============================================================

- Be concise and useful.
- Use headings where useful.
- Use a table when comparing multiple items.
- Separate operational/current information from
  proposed/planned information.
- Preserve exact units such as MMTPA, kTPA, TPD,
  MW, and TPH.
- Every factual claim must have an inline citation.
- Do not add unsupported explanations.

============================================================
USER QUESTION
============================================================

{question}

============================================================
LOCAL KNOWLEDGE BASE
============================================================

{context}

============================================================
FINAL INSTRUCTION
============================================================

Answer ONLY from the retrieved local evidence.
Every factual claim must include an inline source
citation.

If the evidence is insufficient, say so explicitly.
"""

    # ========================================================
    # GENERATE WITH GEMMA
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GENERATING LOCAL GEMMA ANSWER"
    )
    print("=" * 80)

    try:

        answer = ask_gemma(
            prompt
        )

    except Exception as exc:

        print(
            f"Could not generate local answer: "
            f"{exc}"
        )

        return

    if (
        not answer
        or len(answer.strip()) < 20
    ):

        print(
            "WARNING: Gemma returned an "
            "unusually short answer."
        )

        return

    # ========================================================
    # ANSWER
    # ========================================================

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        answer.strip()
    )

    # ========================================================
    # SOURCES
    # ========================================================

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for candidate in selected:

        metadata = candidate[
            "metadata"
        ]

        print(
            f"- "
            f"{metadata.get('source', 'Unknown')} "
            f"(Page "
            f"{metadata.get('page', 'Unknown')})"
        )


# ============================================================
# SHOW COUNT
# ============================================================

def show_count():

    print()

    print(
        f"Knowledge base chunks: "
        f"{collection.count()}"
    )

    print(
        f"Database: "
        f"{CHROMA_DIR}"
    )

    print(
        f"Knowledge base: "
        f"{KB_DIR}"
    )


# ============================================================
# MENU
# ============================================================

def main():

    while True:

        print()
        print("=" * 70)
        print(
            "MRPL SOVEREIGN AI - LOCAL RAG"
        )
        print("=" * 70)

        print()

        print(
            "Local components:"
        )

        print(
            f"  Embeddings : "
            f"{EMBEDDING_MODEL_NAME}"
        )

        print(
            "  Vector DB  : ChromaDB"
        )

        print(
            f"  OCR        : PaddleOCR "
            f"({'available' if PADDLE_AVAILABLE else 'unavailable'})"
        )

        print(
            "  LLM        : Ollama / Gemma 3 4B"
        )

        print(
            f"  KB         : {KB_DIR}"
        )

        print()

        print(
            "1. Build / rebuild knowledge base"
        )

        print(
            "2. Ask a question"
        )

        print(
            "3. Show knowledge base count"
        )

        print(
            "4. Exit"
        )

        print()

        choice = input(
            "Choose an option: "
        ).strip()

        # ----------------------------------------------------
        # Build / rebuild
        # ----------------------------------------------------

        if choice == "1":

            build_index()

        # ----------------------------------------------------
        # Ask question
        # ----------------------------------------------------

        elif choice == "2":

            if collection.count() == 0:

                print()
                print(
                    "The knowledge base is empty."
                )

                print(
                    "Please choose option 1 first."
                )

                continue

            question = input(
                "\nAsk your question: "
            ).strip()

            if question:

                ask_rag(
                    question
                )

            else:

                print(
                    "Question cannot be empty."
                )

        # ----------------------------------------------------
        # Show count
        # ----------------------------------------------------

        elif choice == "3":

            show_count()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        elif choice == "4":

            print(
                "Exiting..."
            )

            break

        else:

            print(
                "Invalid option. "
                "Please choose 1, 2, 3, or 4."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()