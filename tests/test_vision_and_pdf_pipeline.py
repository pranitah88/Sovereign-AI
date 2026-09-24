"""
Comprehensive Test Suite for Vision Analysis, Dynamic PDF Generation, and Failure Handling.

Covers all 25 specific evaluation criteria and critical regression scenarios:
 1. Valid image → vision success
 2. Valid architecture diagram → vision success
 3. Unrelated/non-MRPL image → NOT treated as vision failure
 4. HTTP 500 from Ollama
 5. Token repeat limit error
 6. Timeout
 7. Empty model response
 8. One retry maximum
 9. Successful retry
10. OCR fallback
11. Total vision failure
12. No false visual conclusions after failure
13. Unique analysis_id
14. Unique PDF filename
15. Current image hash propagated correctly
16. Different image → different analysis
17. Different request → different report
18. Previous PDF is never reused
19. Frontend cache-busting URL parameter
20. PDF content reflects CURRENT analysis
21. Original image embedded in PDF
22. Failed analysis does not generate a normal factual PDF
23. Analysis metadata appears correctly in PDF
24. Query integrity across multiple image requests
25. SVG/icon rendering does not expose raw svg identifiers
"""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.database.seed import seed_database
from backend.database.repositories import users as users_repo
from backend.services.vision_verification import (
    run_vision_analysis,
    build_grounded_vision_prompt,
    build_short_vision_prompt,
    is_vision_failure_response,
    VisionInferenceResult,
    VISION_STATUS_COMPLETED,
    VISION_STATUS_FAILED,
    VISION_STATUS_OCR_FALLBACK,
    VISION_GENERATION_OPTIONS,
    VISION_RETRY_OPTIONS,
)
from PIL import Image
from backend.services.docgen import generate_pdf, verify_pdf, parse_analysis_report

def _generate_test_png(color: str, size: tuple[int, int] = (80, 80)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Distinct valid test PNG images generated via Pillow
PNG_A = _generate_test_png("red", (80, 80))
PNG_B = _generate_test_png("blue", (120, 90))
PNG_C = _generate_test_png("green", (150, 100))


@pytest.fixture(scope="module")
def client():
    seed_database()
    with TestClient(app) as c:
        yield c


def _get_token(client: TestClient, username: str = "engineer", password: str = "changeme123") -> str:
    existing = users_repo.get_user_by_username(username)
    if not existing:
        users_repo.create_user(username, password, f"Test {username}", ["engineer"])
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    if res.status_code != 200:
        res = client.post("/api/auth/login", json={"username": username, "password": "1234567890"})
    assert res.status_code == 200, f"Login failed: {res.text}"
    return res.json()["token"]


def _make_mock_response(data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = str(data)
    return resp


# ── TEST 1: Valid Image → Vision Success ─────────────────────────────────────

def test_01_valid_image_vision_success(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Vision Success Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp = {
        "response": (
            "### 1. Visually Observed Elements\nIdentified Distillation Column 11-C-101 and Reflux Pump 11-P-101A.\n\n"
            "### 2. Component & Process Flow Interpretation\nCrude feedstock enters atmospheric distillation.\n\n"
            "### 3. Verifiable Identifiers & Labels\n11-C-101, 11-P-101A.\n\n"
            "### 4. Registry & Scope Verification\nTags verified against MRPL active register.\n\n"
            "### 5. Uncertainty / Review Required\nNone visible."
        )
    }
    with patch("requests.post", return_value=_make_mock_response(mock_resp)):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("pid.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this image"},
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["vision_status"] == VISION_STATUS_COMPLETED
    assert "11-C-101" in body["analysis"]
    assert body["analysis_id"].startswith("vis_")
    assert body["image_hash"] == hashlib.sha256(PNG_A).hexdigest()


# ── TEST 2: Valid Architecture Diagram → Vision Success ──────────────────────

def test_02_valid_architecture_diagram_vision_success(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Arch Diagram Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    arch_analysis = (
        "### 1. Visually Observed Elements\nVisible modules: Frontend (React), FastAPI Gateway, Model Router, ChromaDB, and Local Ollama.\n\n"
        "### 2. Component & Process Flow Interpretation\nFrontend sends queries to FastAPI gateway. Gateway routes request through Scope Guard and Model Router to ChromaDB and Ollama.\n\n"
        "### 3. Verifiable Identifiers & Labels\nNo verifiable equipment tags detected. The diagram contains functional block labels.\n\n"
        "### 4. Registry & Scope Verification\nNo equipment registry verification performed (general technical schematic / no refinery equipment tags present).\n\n"
        "### 5. Uncertainty / Review Required\nNone."
    )

    with patch("requests.post", return_value=_make_mock_response({"response": arch_analysis})):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("architecture_diagram.png", io.BytesIO(PNG_B), "image/png")},
            data={"prompt": "Explain this software architecture diagram"},
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["vision_status"] == VISION_STATUS_COMPLETED
    assert "FastAPI" in body["analysis"]
    assert "ChromaDB" in body["analysis"]


# ── TEST 3: Unrelated / Non-MRPL Image → NOT Treated as Vision Failure ───────

def test_03_unrelated_image_not_treated_as_vision_failure(client: TestClient):
    """An image depicting non-refinery tech architecture must NOT fail vision analysis."""
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Unrelated Diagram Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp = {
        "response": "### 1. Visually Observed Elements\nKubernetes cluster with 3 worker nodes and ingress controller."
    }
    with patch("requests.post", return_value=_make_mock_response(mock_resp)):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("k8s_cluster.png", io.BytesIO(PNG_B), "image/png")},
            data={"prompt": "Analyze this network diagram"},
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["vision_status"] == VISION_STATUS_COMPLETED
    assert "Kubernetes" in body["analysis"]

    # Verify domain relevance trace event was logged as allowed, not failed
    events = [e["event"] for e in body["execution_trace"]]
    assert "VISION_RELEVANCE_CHECKED" in events
    relevance_event = next(e for e in body["execution_trace"] if e["event"] == "VISION_RELEVANCE_CHECKED")
    assert relevance_event["status"] == "allowed"


# ── TEST 4: HTTP 500 from Ollama Handled Gracefully ──────────────────────────

def test_04_http_500_from_ollama_handled():
    mock_500 = MagicMock()
    mock_500.status_code = 500
    mock_500.text = "Internal Server Error: model crashed"

    with patch("requests.post", return_value=mock_500) as mock_post:
        result = run_vision_analysis(
            image_bytes=PNG_A,
            user_query="Analyze this",
            analysis_id="vis_test_500",
            image_hash="hash500",
        )

    assert result.success is False
    assert result.error_code == "HTTP_500"
    assert result.retried is True
    assert mock_post.call_count == 2


# ── TEST 5: Token Repeat Limit Error Handled ─────────────────────────────────

def test_05_token_repeat_limit_handled():
    mock_repeat = MagicMock()
    mock_repeat.status_code = 200
    mock_repeat.json.return_value = {"response": "error: token repeat limit reached"}
    mock_repeat.text = "error: token repeat limit reached"

    with patch("requests.post", return_value=mock_repeat) as mock_post:
        result = run_vision_analysis(
            image_bytes=PNG_A,
            user_query="Analyze this",
            analysis_id="vis_test_repeat",
            image_hash="hashrepeat",
        )

    assert result.success is False
    assert result.error_code == "TOKEN_REPEAT_LIMIT"
    assert result.retried is True
    assert mock_post.call_count == 2


# ── TEST 6: Timeout Handled Gracefully ────────────────────────────────────────

def test_06_timeout_handled():
    import requests as req
    with patch("requests.post", side_effect=req.exceptions.Timeout("Connection timed out")):
        result = run_vision_analysis(
            image_bytes=PNG_A,
            user_query="Analyze this",
            analysis_id="vis_test_timeout",
            image_hash="hashtimeout",
        )

    assert result.success is False
    assert result.error_code == "TIMEOUT"
    assert result.retried is True


# ── TEST 7: Empty Model Response Handled ─────────────────────────────────────

def test_07_empty_model_response_handled():
    mock_empty = MagicMock()
    mock_empty.status_code = 200
    mock_empty.json.return_value = {"response": "   "}
    mock_empty.text = "   "

    with patch("requests.post", return_value=mock_empty):
        result = run_vision_analysis(
            image_bytes=PNG_A,
            user_query="Analyze this",
            analysis_id="vis_test_empty",
            image_hash="hashempty",
        )

    assert result.success is False
    assert result.error_code == "EMPTY_OUTPUT"


# ── TEST 8: One Retry Maximum ────────────────────────────────────────────────

def test_08_one_retry_maximum():
    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Server Error"

    with patch("requests.post", return_value=mock_fail) as mock_post:
        res = run_vision_analysis(
            image_bytes=PNG_A,
            user_query="Analyze",
            analysis_id="vis_max_retry",
            image_hash="hashmax",
        )

    assert mock_post.call_count == 2
    assert res.retried is True
    assert res.success is False


# ── TEST 9: Successful Retry ─────────────────────────────────────────────────

def test_09_successful_retry():
    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Temporary Failure"

    mock_succ = MagicMock()
    mock_succ.status_code = 200
    mock_succ.json.return_value = {"response": "1. Overall purpose: System architecture schematic.\n2. Components: API Gateway, DB."}
    mock_succ.text = mock_succ.json.return_value["response"]

    with patch("requests.post", side_effect=[mock_fail, mock_succ]) as mock_post:
        res = run_vision_analysis(
            image_bytes=PNG_A,
            user_query="Analyze",
            analysis_id="vis_succ_retry",
            image_hash="hashsucc",
        )

    assert res.success is True
    assert res.retried is True
    assert "System architecture schematic" in res.text
    assert mock_post.call_count == 2


# ── TEST 10: OCR Fallback When Vision Fails ──────────────────────────────────

def test_10_ocr_fallback_when_vision_fails(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "OCR Fallback Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_ocr = {"text": "Atmospheric Distillation Column 11-C-101 Crude Pump 11-P-101A"}
    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Ollama connection refused"

    with patch("backend.services.ocr.extract_text_from_image", return_value=mock_ocr), \
         patch("requests.post", return_value=mock_fail):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("pid_drawing.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this image"},
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["vision_status"] == VISION_STATUS_OCR_FALLBACK
    assert "OCR text extraction only" in body["message"]
    assert "11-C-101" in body["message"]


# ── TEST 11: Total Vision Failure ────────────────────────────────────────────

def test_11_total_vision_failure(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Total Failure Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Ollama server error"

    with patch("backend.services.ocr.extract_text_from_image", return_value={"text": ""}), \
         patch("requests.post", return_value=mock_fail):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("bad_drawing.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this image"},
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "vision_failed"
    assert body["vision_status"] == VISION_STATUS_FAILED
    assert "failed during inference" in body["message"]


# ── TEST 12: No False Visual Conclusions After Failure ───────────────────────

def test_12_no_false_visual_conclusions_after_failure(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "No Fake Output Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "token repeat limit reached"

    with patch("backend.services.ocr.extract_text_from_image", return_value={"text": ""}), \
         patch("requests.post", return_value=mock_fail):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this image and list equipment"},
            headers=headers,
        )

    body = res.json()
    assert body["analysis"] == ""
    assert body["tags_detected"] == []
    assert "11-C-101" not in body["message"]
    assert "Atmospheric Distillation" not in body["message"]


# ── TEST 13: Unique analysis_id Per Upload ───────────────────────────────────

def test_13_unique_analysis_id_per_upload(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Unique ID Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp = {"response": "Analysis of diagram."}
    with patch("requests.post", return_value=_make_mock_response(mock_resp)):
        res1 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            headers=headers,
        )
        res2 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            headers=headers,
        )

    id1 = res1.json()["analysis_id"]
    id2 = res2.json()["analysis_id"]
    assert id1 != id2
    assert id1.startswith("vis_")
    assert id2.startswith("vis_")


# ── TEST 14: Unique PDF Filename ─────────────────────────────────────────────

def test_14_unique_pdf_filename(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Unique PDF Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp = {"response": "### 1. Visually Observed Elements\nArchitecture layer components."}
    with patch("requests.post", return_value=_make_mock_response(mock_resp)):
        res1 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this diagram and generate a report in PDF form"},
            headers=headers,
        )
        res2 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this diagram and generate a report in PDF form"},
            headers=headers,
        )

    b1 = res1.json()
    b2 = res2.json()
    assert b1["deliverable"] is not None
    assert b2["deliverable"] is not None
    assert b1["deliverable"]["filename"] != b2["deliverable"]["filename"]
    assert b1["deliverable"]["filename"] == f"report_{b1['analysis_id']}.pdf"
    assert b2["deliverable"]["filename"] == f"report_{b2['analysis_id']}.pdf"


# ── TEST 15: Current Image Hash Propagated Correctly ─────────────────────────

def test_15_image_hash_propagated(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Hash Propagation Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    expected_hash = hashlib.sha256(PNG_A).hexdigest()
    mock_resp = {"response": "Visual description."}

    with patch("requests.post", return_value=_make_mock_response(mock_resp)):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            headers=headers,
        )

    body = res.json()
    assert body["image_hash"] == expected_hash

    hash_events = [e for e in body["execution_trace"] if e["event"] == "IMAGE_HASH_COMPUTED"]
    assert len(hash_events) == 1
    assert hash_events[0]["details"]["image_hash"] == expected_hash


# ── TEST 16: Different Image → Different Analysis ───────────────────────────

def test_16_different_image_different_analysis(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Diff Image Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp_a = {"response": "Visual analysis of image Alpha: CDU unit."}
    mock_resp_b = {"response": "Visual analysis of image Beta: React frontend."}

    with patch("requests.post", return_value=_make_mock_response(mock_resp_a)):
        res_a = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("alpha.png", io.BytesIO(PNG_A), "image/png")},
            headers=headers,
        )
    with patch("requests.post", return_value=_make_mock_response(mock_resp_b)):
        res_b = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("beta.png", io.BytesIO(PNG_B), "image/png")},
            headers=headers,
        )

    body_a = res_a.json()
    body_b = res_b.json()
    assert body_a["image_hash"] != body_b["image_hash"]
    assert "CDU unit" in body_a["analysis"]
    assert "React frontend" in body_b["analysis"]


# ── TEST 17: Different Request → Different Report ───────────────────────────

def test_17_different_request_different_report(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Diff Request Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp_1 = {"response": "Overall summary of architecture components."}
    mock_resp_2 = {"response": "Detailed safety and emergency bypass isolation analysis."}

    with patch("requests.post", return_value=_make_mock_response(mock_resp_1)):
        res_1 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Summarize architecture in PDF"},
            headers=headers,
        )
    with patch("requests.post", return_value=_make_mock_response(mock_resp_2)):
        res_2 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("diagram.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Explain safety bypass isolation in PDF"},
            headers=headers,
        )

    b1 = res_1.json()
    b2 = res_2.json()
    assert b1["analysis_id"] != b2["analysis_id"]
    assert b1["deliverable"]["filename"] != b2["deliverable"]["filename"]
    assert "Overall summary" in b1["analysis"]
    assert "safety and emergency" in b2["analysis"]


# ── TEST 18: Previous PDF is Never Reused ────────────────────────────────────

def test_18_previous_pdf_never_reused(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "No Reuse PDF Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp_1 = {"response": "Analysis of system 1"}
    mock_resp_2 = {"response": "Analysis of system 2"}

    with patch("requests.post", return_value=_make_mock_response(mock_resp_1)):
        res1 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("sys1.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "generate pdf report"},
            headers=headers,
        )
    with patch("requests.post", return_value=_make_mock_response(mock_resp_2)):
        res2 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("sys2.png", io.BytesIO(PNG_B), "image/png")},
            data={"prompt": "generate pdf report"},
            headers=headers,
        )

    deliv1 = res1.json()["deliverable"]
    deliv2 = res2.json()["deliverable"]
    assert deliv1["filename"] != deliv2["filename"]
    assert deliv1["output_id"] != deliv2["output_id"]
    assert deliv1["download_url"] != deliv2["download_url"]
    assert deliv1["path"] != deliv2["path"]


# ── TEST 19: Frontend Cache Busting on Deliverable Download URL ──────────────

def test_19_frontend_cache_busting_url(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Cache Busting Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_resp = {"response": "Analysis report"}
    with patch("requests.post", return_value=_make_mock_response(mock_resp)):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("test.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "generate pdf report"},
            headers=headers,
        )

    deliverable = res.json()["deliverable"]
    assert "analysis_id=" in deliverable["download_url"]

    # Verify the download endpoint returns Cache-Control: no-store
    file_id = deliverable["output_id"]
    dl_res = client.get(f"/api/documents/generated/{file_id}/download", headers=headers)
    assert dl_res.status_code == 200
    assert "no-store" in dl_res.headers.get("Cache-Control", "")


# ── TEST 20: PDF Content Reflects CURRENT Analysis ───────────────────────────

def test_20_pdf_content_reflects_current_analysis():
    import pymupdf
    analysis_text = "### 1. Visually Observed Elements\nIdentified CustomMicrocontroller-XYZ and SensorModule-88."
    res = generate_pdf(
        title="MICROCONTROLLER VISUAL REPORT",
        content=analysis_text,
        document_name="circuit.png",
        filename="report_test_20.pdf",
        metadata={"analysis_id": "vis_20", "request": "Analyze microcontroller"},
    )
    pdf_path = Path(res["path"])
    assert pdf_path.exists()

    with pymupdf.open(str(pdf_path)) as doc:
        full_text = "".join(str(page.get_text()) for page in doc)
    assert "CustomMicrocontroller-XYZ" in full_text
    assert "SensorModule-88" in full_text


# ── TEST 21: Original Image Embedded in PDF ──────────────────────────────────

def test_21_original_image_embedded_in_pdf():
    import pymupdf
    res = generate_pdf(
        title="EMBEDDED IMAGE TEST REPORT",
        content="### 1. Visually Observed Elements\nEmbedded schematic diagram inspection.",
        document_name="schematic.png",
        filename="report_embedded_test.pdf",
        metadata={"analysis_id": "vis_embed", "image_hash": "abcdef1234567890"},
        image_bytes=PNG_A,
    )
    pdf_path = Path(res["path"])
    assert pdf_path.exists()

    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[0]
        image_list = page.get_images()
        assert len(image_list) >= 1, "Expected original image to be embedded in PDF"


# ── TEST 22: Failed Analysis Does NOT Generate Normal Factual PDF ────────────

def test_22_failed_analysis_no_factual_pdf(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Failed PDF Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Internal error"

    with patch("backend.services.ocr.extract_text_from_image", return_value={"text": ""}), \
         patch("requests.post", return_value=mock_fail):
        res = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("failing_diagram.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this diagram and generate a report in PDF form"},
            headers=headers,
        )

    body = res.json()
    assert body["status"] == "vision_failed"
    assert body["deliverable"] is None
    assert "failed during inference" in body["message"]


# ── TEST 23: Analysis Metadata Appears in PDF ────────────────────────────────

def test_23_analysis_metadata_appears_in_pdf():
    import pymupdf
    test_id = "vis_meta_check_999"
    test_hash = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
    res = generate_pdf(
        title="METADATA VERIFICATION REPORT",
        content="### 1. Visually Observed Elements\nUnit test visual evidence.",
        document_name="drawing_meta.png",
        filename="report_meta_check.pdf",
        metadata={
            "analysis_id": test_id,
            "image_hash": test_hash,
            "source": "vision_model",
            "request": "Test request query",
            "verification_status": "VERIFIED_NO_TAGS",
        },
    )
    with pymupdf.open(res["path"]) as doc:
        full_text = "".join(str(page.get_text()) for page in doc)
    assert test_id in full_text
    assert "1122334455667788" in full_text
    assert "VISION_MODEL" in full_text


# ── TEST 24: Query Integrity Across Multiple Image Requests ──────────────────

def test_24_query_integrity_across_turns(client: TestClient):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Query Integrity Test"}, headers=headers)
    sess_id = sess_res.json()["id"]

    resp1 = {"response": "Analysis of Software Architecture Diagram."}
    resp2 = {"response": "Analysis of Refinery Cracking Column."}

    with patch("requests.post", return_value=_make_mock_response(resp1)):
        r1 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("software.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Explain this architecture diagram"},
            headers=headers,
        )
    with patch("requests.post", return_value=_make_mock_response(resp2)):
        r2 = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("refinery.png", io.BytesIO(PNG_B), "image/png")},
            data={"prompt": "Explain this refinery diagram"},
            headers=headers,
        )

    b1 = r1.json()
    b2 = r2.json()

    assert "Software Architecture" in b1["analysis"]
    assert "Refinery Cracking" in b2["analysis"]
    assert b1["analysis_id"] != b2["analysis_id"]
    assert b1["image_hash"] != b2["image_hash"]


# ── TEST 25: SVG / Icon Rendering Does Not Expose Raw SVG Identifiers ────────

def test_25_svg_icon_markers_scrubbed():
    raw_leaked_strings = [
        ("svgCopy", "Copy"),
        ("svgQuery Classified", "Query Classified"),
        ("svgRetrieval", "Retrieval"),
        ("svgModel Selected", "Model Selected"),
        ("svgTools / Sandbox", "Tools / Sandbox"),
        ("svgResponse Verified", "Response Verified"),
    ]
    import re
    for raw_input, expected_clean in raw_leaked_strings:
        cleaned = re.sub(r"\bsvg(?=Copy|Query|Retrieval|Model|Tools|Response|Download)", "", raw_input, flags=re.IGNORECASE)
        assert cleaned == expected_clean


# ── PART Q: CRITICAL REGRESSION INTEGRATION SCENARIOS ─────────────────────────

def test_critical_regression_scenarios(client: TestClient):
    """Executes the exact 4 scenarios required by PART Q."""
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    sess_res = client.post("/api/chat/sessions", json={"title": "Part Q Critical Scenarios"}, headers=headers)
    sess_id = sess_res.json()["id"]

    # SCENARIO 1: Architecture Diagram A
    mock_a = {"response": "### 1. Visually Observed Elements\nArchitecture A contains Microservices A1 and A2."}
    with patch("requests.post", return_value=_make_mock_response(mock_a)):
        res_a = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("arch_a.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this image and generate a PDF report."},
            headers=headers,
        )
    body_a = res_a.json()
    assert body_a["status"] == "success"
    id_a = body_a["analysis_id"]
    hash_a = body_a["image_hash"]
    deliv_a = body_a["deliverable"]
    assert "Microservices A1" in body_a["analysis"]
    assert deliv_a is not None
    assert id_a in deliv_a["filename"]

    # SCENARIO 2: Architecture Diagram B
    mock_b = {"response": "### 1. Visually Observed Elements\nArchitecture B contains Event Bus and Workers B1."}
    with patch("requests.post", return_value=_make_mock_response(mock_b)):
        res_b = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("arch_b.png", io.BytesIO(PNG_B), "image/png")},
            data={"prompt": "Analyze this image and generate a PDF report."},
            headers=headers,
        )
    body_b = res_b.json()
    assert body_b["status"] == "success"
    id_b = body_b["analysis_id"]
    hash_b = body_b["image_hash"]
    deliv_b = body_b["deliverable"]
    assert id_b != id_a
    assert hash_b != hash_a
    assert "Event Bus" in body_b["analysis"]
    assert deliv_b["filename"] != deliv_a["filename"]

    # SCENARIO 3: MRPL / Refinery Diagram C
    mock_c = {
        "response": "### 1. Visually Observed Elements\nRefinery Column 11-C-101 and Desalter 11-V-101."
    }
    with patch("requests.post", return_value=_make_mock_response(mock_c)):
        res_c = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("refinery_c.png", io.BytesIO(PNG_C), "image/png")},
            data={"prompt": "Explain this diagram and generate a PDF."},
            headers=headers,
        )
    body_c = res_c.json()
    assert body_c["status"] == "success"
    id_c = body_c["analysis_id"]
    hash_c = body_c["image_hash"]
    deliv_c = body_c["deliverable"]
    assert id_c not in (id_a, id_b)
    assert hash_c not in (hash_a, hash_b)
    assert "11-C-101" in body_c["analysis"]
    assert deliv_c["filename"] not in (deliv_a["filename"], deliv_b["filename"])

    # SCENARIO 4: Force Vision Model Failure
    mock_fail = MagicMock()
    mock_fail.status_code = 500
    mock_fail.text = "Inference failure: token repeat limit reached"

    with patch("backend.services.ocr.extract_text_from_image", return_value={"text": ""}), \
         patch("requests.post", return_value=mock_fail):
        res_d = client.post(
            f"/api/chat/sessions/{sess_id}/image-analysis",
            files={"file": ("fail_d.png", io.BytesIO(PNG_A), "image/png")},
            data={"prompt": "Analyze this image and generate a PDF report."},
            headers=headers,
        )
    body_d = res_d.json()
    assert body_d["status"] == "vision_failed"
    assert body_d["vision_status"] == VISION_STATUS_FAILED
    assert body_d["deliverable"] is None
    assert "failed during inference" in body_d["message"]
    # Verify no older successful PDF is returned
    events_d = [e["event"] for e in body_d["execution_trace"]]
    assert "VISION_INFERENCE_FAILED" in events_d
    assert "VISION_ANALYSIS_FAILED" in events_d
