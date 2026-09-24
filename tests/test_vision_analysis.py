"""
Comprehensive Test Suite for Image Upload & Local Sovereign Vision Analysis.

Covers all 10 mandatory evaluation points:
1. PNG upload
2. JPG upload
3. Invalid file rejection
4. Oversized image rejection
5. Viewer / RBAC behavior
6. Local Ollama vision inference
7. P&ID tag extraction
8. Unknown tag detection & escalation to Human Approval
9. Network Seal remains satisfied (zero cloud egress)
10. Existing text/RAG/coding workflows still pass, and temp image files are deleted.
"""

import io
import os
import pathlib
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app import app
from backend.database.seed import seed_database
from backend.database.repositories import users as users_repo
from backend.database.repositories import chat as chat_repo
from backend.database.repositories import approvals as approvals_repo
from backend.database.repositories import audit as audit_repo
from backend.services.network_seal import get_network_seal
from backend.services.task_router import select_vision_model, is_ollama_model_installed
from backend.services.vision_verification import (
    extract_pid_tags_from_text,
    validate_equipment_tags,
    verify_pid_drawing,
    extract_process_labels,
    extract_grounded_equipment_tags,
    is_tag_grounded_in_visual_evidence,
    is_vision_failure_response,
    run_vision_analysis,
    build_short_vision_prompt,
    VisionInferenceResult,
    KNOWN_EQUIPMENT_REGISTER,
    VISION_STATUS_COMPLETED,
    VISION_STATUS_FAILED,
    VISION_STATUS_OCR_FALLBACK,
    VISION_GENERATION_OPTIONS,
    VISION_RETRY_OPTIONS,
)


@pytest.fixture(scope="module")
def client():
    seed_database()
    with TestClient(app) as c:
        yield c


def _get_token(client: TestClient, username: str = "engineer", password: str = "changeme123") -> str:
    # Check if user exists; if not, create
    existing = users_repo.get_user_by_username(username)
    if not existing:
        role = "viewer" if "viewer" in username else "engineer"
        users_repo.create_user(username, password, f"Test {username}", [role])
    else:
        # Try primary test password
        res = client.post("/api/auth/login", json={"username": username, "password": password})
        if res.status_code == 200:
            return res.json()["token"]
        # Try seed demo password
        res_demo = client.post("/api/auth/login", json={"username": username, "password": "1234567890"})
        if res_demo.status_code == 200:
            return res_demo.json()["token"]

    res = client.post("/api/auth/login", json={"username": username, "password": password})
    if res.status_code != 200:
        pytest.fail(f"Login failed for {username}: {res.text}")
    return res.json()["token"]


def _make_mock_response(data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = str(data)
    return resp


# Minimal 1x1 valid PNG bytes
VALID_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal valid JPEG bytes
VALID_JPG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00"
    b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f"
    b"\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0"
    b"\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01"
    b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda"
    b"\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)


# ── 1. PNG Upload ─────────────────────────────────────────────────────────────

def test_01_png_upload(client: TestClient):
    """Test valid PNG upload and analysis through the FastAPI endpoint."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    # Create session
    session_res = client.post("/api/chat/sessions", json={"title": "Vision PNG Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_ollama_resp = {
        "response": "Detailed analysis of P&ID: Identified Crude Distillation Tower 11-C-101 and transfer pump 11-P-101A. Operating pressure is within normal design envelope."
    }

    with patch("requests.post", return_value=_make_mock_response(mock_ollama_resp)):
        files = {"file": ("pid_drawing.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "Analyze this P&ID diagram"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert "11-C-101" in body["analysis"]
    assert "11-P-101A" in body["analysis"]
    assert body["verification"]["status"] == "VERIFIED"
    assert body["verification"]["is_approved"] is True
    assert "11-C-101" in body["verification"]["equipment_tags"]

    # Check execution trace events
    events = [e["event"] for e in body["execution_trace"]]
    assert "IMAGE_RECEIVED" in events
    assert "FILE_VALIDATED" in events
    assert "RBAC_CHECK" in events
    assert "VISION_MODEL_SELECTED" in events
    assert "OLLAMA_INFERENCE" in events
    assert "VISION_ANALYSIS_COMPLETED" in events
    assert "VISION_VERIFICATION_COMPLETED" in events
    assert "AUDIT" in events


# ── 2. JPG Upload ─────────────────────────────────────────────────────────────

def test_02_jpg_upload(client: TestClient):
    """Test valid JPG upload and analysis through the FastAPI endpoint."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Vision JPG Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_ollama_resp = {
        "response": "Inspection photograph indicates primary pump 11-P-101A casing and mechanical seals are intact."
    }

    with patch("requests.post", return_value=_make_mock_response(mock_ollama_resp)):
        files = {"file": ("equipment_photo.jpg", io.BytesIO(VALID_JPG_BYTES), "image/jpeg")}
        data = {"prompt": "Inspect pump seal"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert "11-P-101A" in body["verification"]["equipment_tags"]


# ── 3. Invalid File Rejection ─────────────────────────────────────────────────

def test_03_invalid_file_rejection(client: TestClient):
    """Test rejection of non-image files (.txt, .exe) and fraudulent mime types."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Invalid File Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    # Test 3a: Unsupported extension .txt
    files_txt = {"file": ("malicious.txt", io.BytesIO(b"Hello world text"), "text/plain")}
    res_txt = client.post(
        f"/api/chat/sessions/{session_id}/image-analysis",
        files=files_txt,
        headers=headers,
    )
    assert res_txt.status_code == 400
    assert "Unsupported file extension" in res_txt.json()["detail"]

    # Test 3b: Spoofed extension with non-image mime
    files_spoof = {"file": ("script.png", io.BytesIO(b"import os; os.system('calc')"), "application/x-executable")}
    res_spoof = client.post(
        f"/api/chat/sessions/{session_id}/image-analysis",
        files=files_spoof,
        headers=headers,
    )
    assert res_spoof.status_code == 400
    assert "Unsupported MIME type" in res_spoof.json()["detail"]


# ── 4. Oversized Image Rejection ──────────────────────────────────────────────

def test_04_oversized_image_rejection(client: TestClient):
    """Test rejection of images exceeding the 10 MB strict limit."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Oversized Image Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    # 10 MB + 1024 bytes
    oversized_data = b"\x89PNG\r\n\x1a\n" + (b"0" * (10 * 1024 * 1024 + 1024))
    files = {"file": ("large_drawing.png", io.BytesIO(oversized_data), "image/png")}
    res = client.post(
        f"/api/chat/sessions/{session_id}/image-analysis",
        files=files,
        headers=headers,
    )
    assert res.status_code == 413
    assert "10 MB limit" in res.json()["detail"]


# ── 5. Viewer / RBAC Behavior ─────────────────────────────────────────────────

def test_05_viewer_and_rbac_behavior(client: TestClient):
    """Verify viewer can analyze images within their clearance, but unauthorized users are blocked."""
    viewer_token = _get_token(client, "viewer")
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

    # 5a. Viewer can create a session and analyze an unclassified/routine image
    sess_res = client.post("/api/chat/sessions", json={"title": "Viewer Vision Session"}, headers=viewer_headers)
    assert sess_res.status_code in [200, 201]
    session_id = sess_res.json()["id"]

    mock_ollama_resp = {"response": "P&ID contains 11-C-101 and 11-P-101A."}
    with patch("requests.post", return_value=_make_mock_response(mock_ollama_resp)):
        files = {"file": ("pid.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            headers=viewer_headers,
        )
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 5b. Unauthenticated access rejected with 401
    client.cookies.clear()
    files_noauth = {"file": ("pid.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
    res_noauth = client.post(f"/api/chat/sessions/{session_id}/image-analysis", files=files_noauth)
    assert res_noauth.status_code == 401

    # 5c. User cannot upload to another user's session
    admin_token = _get_token(client, "admin")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    admin_sess = client.post("/api/chat/sessions", json={"title": "Admin Private Session"}, headers=admin_headers).json()

    files_cross = {"file": ("pid.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
    res_cross = client.post(
        f"/api/chat/sessions/{admin_sess['id']}/image-analysis",
        files=files_cross,
        headers=viewer_headers,
    )
    assert res_cross.status_code in [403, 404]  # Isolated per user


# ── 6. Local Ollama Vision Inference ──────────────────────────────────────────

def test_06_local_ollama_vision_inference():
    """Verify select_vision_model dynamically checks installed models and selects local Ollama model."""
    model_entry = select_vision_model()
    ollama_model = model_entry.get("ollama_model_name", "")
    assert ollama_model in ["qwen2.5vl:3b", "qwen2.5-vl:3b", "gemma3:4b"]
    assert "gpt" not in ollama_model.lower()
    assert "claude" not in ollama_model.lower()
    assert "gemini" not in ollama_model.lower()


# ── 7. P&ID Tag Extraction ────────────────────────────────────────────────────

def test_07_pid_tag_extraction():
    """Test deterministic regex extraction of equipment tags from model output."""
    sample_text = """
    P&ID Analysis:
    - Primary Distillation Column: 11-C-101
    - Reflux Pump: 11-P-101A and spare 11-P-101B
    - Reactor Vessel: 21-R-101
    - Heavy Gas Oil Pump: 31-P-101A
    """
    extracted = extract_pid_tags_from_text(sample_text)
    tags = extracted["equipment_tags"]
    assert "11-C-101" in tags
    assert "11-P-101A" in tags
    assert "11-P-101B" in tags
    assert "21-R-101" in tags
    assert "31-P-101A" in tags

    validation = validate_equipment_tags(tags)
    assert validation.is_approved is True
    assert validation.requires_human_review is False
    assert len(validation.mismatched_tags) == 0


# ── 8. Unknown Tag Detection & Human Review Escalation ─────────────────────────

def test_08_unknown_tag_detection_and_escalation(client: TestClient):
    """Verify unknown / uncatalogued equipment tags trigger REQUIRES REVIEW and approval queue."""
    sample_text_with_unknown = """
    Identified Equipment:
    - Tower 11-C-101 (Verified)
    - Uncatalogued high-pressure bypass valve: 99-X-999
    - Unknown auxiliary pump: 88-P-001
    """
    extracted = extract_pid_tags_from_text(sample_text_with_unknown)
    tags = extracted["equipment_tags"]
    assert "11-C-101" in tags
    assert "99-X-999" in tags
    assert "88-P-001" in tags

    validation = validate_equipment_tags(tags)
    assert validation.is_approved is False
    assert validation.requires_human_review is True
    assert "99-X-999" in validation.mismatched_tags
    assert "88-P-001" in validation.mismatched_tags

    # Now verify via image-analysis endpoint that unknown tag creates approval proposal
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}
    session_res = client.post("/api/chat/sessions", json={"title": "Unknown Tag Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_ollama_resp = {
        "response": "Drawing contains tower 11-C-101 and unregistered booster 99-X-999."
    }
    with patch("requests.post", return_value=_make_mock_response(mock_ollama_resp)):
        files = {"file": ("pid_unknown.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()
    assert body["verification"]["status"] == "REQUIRES_REVIEW"
    assert body["verification"]["is_approved"] is False
    assert body["verification"]["requires_human_review"] is True
    assert "99-X-999" in body["verification"]["unknown_tags"]
    assert body["approval_id"] is not None

    # Verify in approvals repository
    pending = approvals_repo.list_pending_approvals()
    proposal_ids = [p["id"] for p in pending]
    assert body["approval_id"] in proposal_ids


# ── 9. Network Seal Remains Satisfied (Zero Cloud Egress) ─────────────────────

def test_09_network_seal_remains_satisfied():
    """Verify Network Seal strictly blocks external egress and enforces local loopback."""
    seal = get_network_seal()

    # 9a. Local addresses permitted
    assert seal.is_local_address("http://127.0.0.1:11434/api/generate") is True
    assert seal.is_local_address("http://localhost:11434") is True
    assert seal.record_local_call("http://127.0.0.1:11434/api/generate", "Ollama Vision") is True

    # 9b. External addresses strictly blocked with audited exception
    initial_blocked = seal.blocked_external_attempts
    with pytest.raises(PermissionError):
        seal.enforce_no_egress("https://api.openai.com/v1/images/generations", "Cloud Vision Blocked")
    with pytest.raises(PermissionError):
        seal.enforce_no_egress("https://generativelanguage.googleapis.com/v1/models", "Cloud Vision Blocked")
    with pytest.raises(PermissionError):
        seal.enforce_no_egress("https://api.anthropic.com/v1/messages", "Cloud Vision Blocked")

    assert seal.blocked_external_attempts >= initial_blocked + 3


# ── 10. Existing Workflows Still Pass & Temp Images Cleaned Up ────────────────

def test_10_existing_workflows_and_cleanup(client: TestClient):
    """Verify standard text chat, session persistence, and guaranteed cleanup of temp images."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    # 10a. Existing text message chat still works
    session_res = client.post("/api/chat/sessions", json={"title": "Text Regression Session"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    msg_res = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": "What is the design capacity of the MRPL CDU unit?"},
        headers=headers,
    )
    assert msg_res.status_code == 201
    assert msg_res.json()["role"] == "user"

    # 10b. Verify temp_images directory does not retain temporary image files
    temp_dir = pathlib.Path("data/temp_images")
    if temp_dir.exists():
        temp_files = list(temp_dir.glob("*"))
        # Any file created during the test run should have been cleaned up
        assert len(temp_files) == 0, f"Leaked temporary image files found: {temp_files}"


# ── 11. Refinery Diagram Vision Regression (Zero Hallucination) ────────────────

def test_11_refinery_diagram_regression(client: TestClient):
    """
    Regression Test for uninstrumented refinery diagram (tests/fixtures/refinery.png):
    - Process labels detected: Atmospheric Dist., Vacuum Dist., Fluid Cat. Cracking, Vis-breaking, Furnace, etc.
    - No verified equipment tag IDs
    - No equipment registry verification
    - No false unknown-tag human review (requires_human_review is False, approval_id is None)
    - No invented MRPL attribution
    - No unsupported compliance claim
    - Hallucinated tags like 11-P-101A, 11-P-101B, 11-P-102A, 11-P-102B, 11-P-103A strictly rejected
    """
    fixture_path = pathlib.Path("tests/fixtures/refinery.png")
    assert fixture_path.exists(), f"Refinery test fixture missing: {fixture_path}"

    # 1. Test deterministic python verification service directly
    report = verify_pid_drawing(fixture_path)
    assert report.status == "NO_VERIFIABLE_TAGS"
    assert report.human_approval_id is None
    assert len(report.extracted_tags) == 0
    assert len(report.matched_tags) == 0
    assert len(report.unregistered_tags) == 0
    assert "No verifiable equipment tags detected" in report.summary

    # Verify detected process labels
    assert "Atmospheric Dist." in report.process_labels
    assert "Vacuum Dist." in report.process_labels
    assert "Fluid Cat. Cracking" in report.process_labels
    assert "Vis-breaking" in report.process_labels
    assert "Furnace" in report.process_labels

    # Verify forbidden hallucinated tags are not present
    hallucinated_tags = ["11-P-101A", "11-P-101B", "11-P-102A", "11-P-102B", "11-P-103A"]
    for ht in hallucinated_tags:
        assert ht not in report.extracted_tags

    # 2. Test full FastAPI image analysis endpoint with real refinery image
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}
    session_res = client.post("/api/chat/sessions", json={"title": "Refinery Diagram Regression"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    with open(fixture_path, "rb") as f:
        img_bytes = f.read()

    files = {"file": ("refinery.png", io.BytesIO(img_bytes), "image/png")}
    res = client.post(
        f"/api/chat/sessions/{session_id}/image-analysis",
        files=files,
        data={"query": "Analyze this refinery flow diagram."},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()

    # Verify endpoint results
    verif = data["verification"]
    assert verif["status"] == "NO_VERIFIABLE_TAGS"
    assert verif["is_approved"] is True
    assert verif["requires_human_review"] is False
    assert verif["approval_id"] is None
    assert len(verif["equipment_tags"]) == 0
    assert len(verif["matched_tags"]) == 0
    assert len(verif["unknown_tags"]) == 0

    # Ensure hallucinated tags are NOT in detected tags
    for ht in hallucinated_tags:
        assert ht not in verif["equipment_tags"]

    # Verify detected process labels in API response
    proc_labels = verif.get("process_labels", [])
    assert any("Atmospheric Dist" in p for p in proc_labels)
    assert any("Vacuum Dist" in p for p in proc_labels)
    assert any("Fluid Cat. Cracking" in p for p in proc_labels)
    assert any("Vis-breaking" in p for p in proc_labels)
    assert any("Furnace" in p for p in proc_labels)

    # Verify the 5 structured sections in analysis output
    analysis = data["analysis"]
    assert "### 1. Visually Observed Elements" in analysis
    assert "### 2. Process Interpretation" in analysis
    assert "### 3. Verifiable Equipment Tags" in analysis
    assert "### 4. Registry Verification" in analysis
    assert "### 5. Uncertainty / Review Required" in analysis

    # Verify absence of false unknown tag review proposal in DB
    pending = approvals_repo.list_pending_approvals()
    drawing_approvals = [
        p for p in pending
        if "refinery.png" in str(p.get("affected_resource", "")) or "refinery.png" in str(p.get("proposed_payload", ""))
    ]
    assert len(drawing_approvals) == 0, f"False approval proposal created: {drawing_approvals}"


# ═══════════════════════════════════════════════════════════════════════════════
# VISION FAILURE HANDLING TESTS (12-29)
# Tests for fail-closed behavior when vision model returns errors.
# ═══════════════════════════════════════════════════════════════════════════════


# ── 12. HTTP 500 from Vision Model ────────────────────────────────────────────

def test_12_vision_model_http_500(client: TestClient):
    """
    When qwen2.5-vl:3b returns HTTP 500, the endpoint must:
    - NOT return status="success"
    - NOT claim "No Verifiable Equipment Tags"
    - NOT produce process label claims
    - Return a controlled failure message
    """
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "HTTP 500 Vision Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    # Mock Ollama returning HTTP 500
    mock_500 = _make_mock_response(
        {"error": "prediction aborted, token repeat limit reached"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_500):
        files = {"file": ("diagram.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this diagram"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    assert res.status_code == 200
    body = res.json()

    # FAIL CLOSED assertions
    assert body["status"] == "vision_failed"
    assert body["vision_status"] == VISION_STATUS_FAILED
    assert body["verification"]["status"] == "VISION_ANALYSIS_FAILED"
    assert body["verification"]["is_approved"] is False
    assert len(body["verification"]["equipment_tags"]) == 0
    assert len(body["verification"]["process_labels"]) == 0

    # Must NOT contain fabricated claims
    msg = body["message"]
    assert "No Verifiable Equipment Tags" not in msg
    assert "Detected Process Labels" not in msg
    assert "Response Verified" not in msg

    # Must contain controlled failure message
    assert "Visual analysis could not be completed" in msg
    assert "No visual conclusions were made" in msg

    # Raw error must NOT be exposed to user
    assert body["vision_error"] == ""


# ── 13. Token Repeat Limit Failure ────────────────────────────────────────────

def test_13_token_repeat_limit_failure(client: TestClient):
    """
    Exact reproduction of the reported bug:
    qwen2.5-vl:3b HTTP 500 'prediction aborted, token repeat limit reached'
    Must return VISION_ANALYSIS_FAILED, not 'No Verifiable Equipment Tags'.
    """
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Token Repeat Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_error = _make_mock_response(
        {"error": "prediction aborted, token repeat limit reached"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_error):
        files = {"file": ("refinery_flow.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this diagram"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert body["vision_status"] == VISION_STATUS_FAILED
    assert body["verification"]["status"] == "VISION_ANALYSIS_FAILED"
    assert "No Verifiable Equipment Tags" not in body["message"]
    assert body["analysis"] == ""


# ── 14. Timeout from Vision Model ─────────────────────────────────────────────

def test_14_vision_model_timeout(client: TestClient):
    """Vision model timeout must result in VISION_ANALYSIS_FAILED."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Timeout Vision Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    import requests as req_mod
    with patch("requests.post", side_effect=req_mod.exceptions.Timeout("Connection timed out")):
        files = {"file": ("timeout.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "analyze this"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert body["vision_status"] == VISION_STATUS_FAILED
    assert body["verification"]["status"] == "VISION_ANALYSIS_FAILED"
    assert "Visual analysis could not be completed" in body["message"]


# ── 15. Empty Output from Vision Model ────────────────────────────────────────

def test_15_vision_model_empty_output(client: TestClient):
    """Vision model returning empty string must be detected as failure."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Empty Output Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_empty = _make_mock_response({"response": ""}, status_code=200)

    with patch("requests.post", return_value=mock_empty):
        files = {"file": ("empty.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert body["vision_status"] == VISION_STATUS_FAILED
    assert body["verification"]["status"] == "VISION_ANALYSIS_FAILED"


# ── 16. Exactly One Retry Only ────────────────────────────────────────────────

def test_16_exactly_one_retry():
    """run_vision_analysis must attempt exactly ONE retry, not unlimited."""
    # Track calls to requests.post
    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_mock_response(
            {"error": "prediction aborted, token repeat limit reached"},
            status_code=500,
        )

    with patch("requests.post", side_effect=mock_post):
        with patch("backend.services.vision_verification.record_local_call"):
            result = run_vision_analysis(
                image_bytes=VALID_PNG_BYTES,
                user_query="explain this diagram",
            )

    assert result.success is False
    assert result.retried is True
    # Exactly 2 calls: initial + 1 retry
    assert call_count == 2


# ── 17. Retry Failure Stops ───────────────────────────────────────────────────

def test_17_retry_failure_stops():
    """When both initial and retry fail, VISION_ANALYSIS_FAILED is returned."""
    mock_fail = _make_mock_response(
        {"error": "token repeat limit reached"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_fail):
        with patch("backend.services.vision_verification.record_local_call"):
            result = run_vision_analysis(
                image_bytes=VALID_PNG_BYTES,
                user_query="analyze this",
            )

    assert result.success is False
    assert result.retried is True
    assert result.error_code in ["TOKEN_REPEAT_LIMIT", "HTTP_500"]
    assert result.source == "none"


# ── 18. Successful Retry ──────────────────────────────────────────────────────

def test_18_successful_retry():
    """If initial attempt fails but retry succeeds, result should be success."""
    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call fails
            return _make_mock_response(
                {"error": "prediction aborted"},
                status_code=500,
            )
        else:
            # Retry succeeds
            return _make_mock_response(
                {"response": "This diagram shows a crude distillation process."},
                status_code=200,
            )

    with patch("requests.post", side_effect=mock_post):
        with patch("backend.services.vision_verification.record_local_call"):
            result = run_vision_analysis(
                image_bytes=VALID_PNG_BYTES,
                user_query="explain this diagram",
            )

    assert result.success is True
    assert result.retried is True
    assert "crude distillation" in result.text.lower()
    assert call_count == 2


# ── 19. OCR Fallback After Vision Failure ─────────────────────────────────────

def test_19_ocr_fallback_on_vision_failure(client: TestClient):
    """When vision fails but OCR succeeds, response is OCR-derived."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "OCR Fallback Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    # Mock: vision model fails, but OCR returns text
    mock_500 = _make_mock_response(
        {"error": "prediction aborted, token repeat limit reached"},
        status_code=500,
    )

    ocr_mock_result = {
        "text": "Crude Oil\nFurnace\nAtmospheric Distillation\nGas Oil",
        "confidence": 0.92,
        "line_count": 4,
        "lines": [],
    }

    with patch("requests.post", return_value=mock_500):
        with patch(
            "backend.services.ocr.extract_text_from_image",
            return_value=ocr_mock_result,
        ):
            files = {"file": ("ocr_test.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
            data = {"prompt": "explain this diagram"}
            res = client.post(
                f"/api/chat/sessions/{session_id}/image-analysis",
                files=files,
                data=data,
                headers=headers,
            )

    body = res.json()
    assert body["vision_status"] == VISION_STATUS_OCR_FALLBACK
    # OCR fallback response must indicate OCR-derived
    assert "OCR" in body["message"]
    assert "visual understanding" not in body["message"].lower() or "do not represent full visual understanding" in body["message"].lower()

    # Trace must include OCR fallback event
    events = [e["event"] for e in body["execution_trace"]]
    assert "VISION_OCR_FALLBACK" in events
    assert "VISION_ANALYSIS_FAILED" not in events  # OCR fallback is NOT full failure


# ── 20. Both Vision and OCR Fail ──────────────────────────────────────────────

def test_20_both_vision_and_ocr_fail(client: TestClient):
    """When both vision and OCR fail, controlled failure response."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Both Fail Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_500 = _make_mock_response(
        {"error": "prediction aborted"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_500):
        with patch(
            "backend.services.ocr.extract_text_from_image",
            side_effect=ImportError("PaddleOCR not available"),
        ):
            files = {"file": ("both_fail.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
            data = {"prompt": "explain this diagram"}
            res = client.post(
                f"/api/chat/sessions/{session_id}/image-analysis",
                files=files,
                data=data,
                headers=headers,
            )

    body = res.json()
    assert body["status"] == "vision_failed"
    assert body["vision_status"] == VISION_STATUS_FAILED
    assert "Visual analysis could not be completed" in body["message"]
    assert body["analysis"] == ""

    # Trace must include VISION_ANALYSIS_FAILED
    events = [e["event"] for e in body["execution_trace"]]
    assert "VISION_ANALYSIS_FAILED" in events


# ── 21. Successful Vision With No Tags (VERIFIED_NO_TAGS) ─────────────────────

def test_21_successful_vision_no_tags(client: TestClient):
    """
    When vision succeeds and genuinely finds no equipment tags, the status
    must be VERIFIED_NO_TAGS (not VISION_ANALYSIS_FAILED).
    """
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "No Tags Success Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    # Vision model succeeds with a valid analysis containing no equipment tags
    mock_success = _make_mock_response(
        {"response": "This diagram shows a generic process flow. No specific equipment tags are visible."},
        status_code=200,
    )

    with patch("requests.post", return_value=mock_success):
        files = {"file": ("notags.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this diagram"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert body["status"] == "success"
    assert body["vision_status"] == VISION_STATUS_COMPLETED
    assert body["verification"]["status"] == "NO_VERIFIABLE_TAGS"
    assert body["verification"]["is_approved"] is True
    # This is NOT a failed analysis
    assert "Visual analysis could not be completed" not in body["message"]


# ── 22. Successful Vision With Valid Tags ─────────────────────────────────────

def test_22_successful_vision_with_valid_tags(client: TestClient):
    """Vision succeeds and finds valid registered equipment tags → VERIFIED."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Valid Tags Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_success = _make_mock_response(
        {"response": "P&ID shows Crude Distillation Tower 11-C-101 and pump 11-P-101A in service."},
        status_code=200,
    )

    with patch("requests.post", return_value=mock_success):
        files = {"file": ("tags.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "analyze this P&ID"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert body["status"] == "success"
    assert body["vision_status"] == VISION_STATUS_COMPLETED
    assert body["verification"]["status"] == "VERIFIED"
    assert body["verification"]["is_approved"] is True
    assert "11-C-101" in body["verification"]["equipment_tags"]
    assert "11-P-101A" in body["verification"]["equipment_tags"]


# ── 23. Failed Vision Does NOT Produce "No Verifiable Equipment Tags" ─────────

def test_23_failed_vision_no_verifiable_tags_claim(client: TestClient):
    """
    CRITICAL: A failed vision analysis must NEVER produce the string
    'No Verifiable Equipment Tags' — that implies successful analysis.
    """
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "No False Tags Claim"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_fail = _make_mock_response(
        {"error": "prediction aborted, token repeat limit reached"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_fail):
        files = {"file": ("fail.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this diagram"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    msg = body["message"]
    assert "No Verifiable Equipment Tags" not in msg
    assert body["verification"]["status"] != "NO_VERIFIABLE_TAGS"
    assert body["verification"]["status"] == "VISION_ANALYSIS_FAILED"


# ── 24. Failed Vision Does NOT Produce "Detected Process Labels" ──────────────

def test_24_failed_vision_no_process_labels_claim(client: TestClient):
    """
    Failed vision analysis must NOT produce 'Detected Process Labels: ...'
    unless those labels came from a verified OCR fallback.
    """
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "No False Labels"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_fail = _make_mock_response(
        {"error": "prediction aborted"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_fail):
        files = {"file": ("nolabels.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "analyze"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert "Detected Process Labels" not in body["message"]
    # Process labels list must be empty when vision failed
    assert len(body["verification"]["process_labels"]) == 0


# ── 25. Failed Vision Does NOT Produce "Response Verified" ────────────────────

def test_25_failed_vision_no_response_verified(client: TestClient):
    """Failed vision analysis must NOT end with 'Response Verified' or similar."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "No Response Verified"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_fail = _make_mock_response(
        {"error": "prediction aborted, token repeat limit reached"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_fail):
        files = {"file": ("novrfy.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    assert "Response Verified" not in body["message"]

    # Trace must NOT contain VISION_VERIFICATION_COMPLETED
    events = [e["event"] for e in body["execution_trace"]]
    assert "VISION_VERIFICATION_COMPLETED" not in events


# ── 26. Execution Trace for Failure ───────────────────────────────────────────

def test_26_execution_trace_failure(client: TestClient):
    """Verify failure execution trace contains expected events in order."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Trace Failure Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_fail = _make_mock_response(
        {"error": "prediction aborted, token repeat limit reached"},
        status_code=500,
    )

    with patch("requests.post", return_value=mock_fail):
        files = {"file": ("trace_fail.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "explain this diagram"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    events = [e["event"] for e in body["execution_trace"]]

    # Required events for failure
    assert "VISION_MODEL_SELECTED" in events
    assert "VISION_ANALYSIS_STARTED" in events
    assert "VISION_INFERENCE_FAILED" in events
    assert "VISION_ANALYSIS_FAILED" in events
    assert "AUDIT" in events

    # Must NOT have successful verification events
    assert "VISION_VERIFICATION_COMPLETED" not in events
    assert "VISION_ANALYSIS_COMPLETED" not in events

    # Check that the inference failure has error details preserved
    failed_events = [e for e in body["execution_trace"] if e["event"] == "VISION_INFERENCE_FAILED"]
    assert len(failed_events) > 0
    for fe in failed_events:
        details = fe.get("details", {})
        # Error details must be preserved internally
        assert details.get("error_code") or details.get("error")


# ── 27. Execution Trace for Success ───────────────────────────────────────────

def test_27_execution_trace_success(client: TestClient):
    """Verify success execution trace contains expected events."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Trace Success Test"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_success = _make_mock_response(
        {"response": "P&ID shows Atmospheric Distillation Column 11-C-101."},
        status_code=200,
    )

    with patch("requests.post", return_value=mock_success):
        files = {"file": ("trace_ok.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        data = {"prompt": "analyze this P&ID"}
        res = client.post(
            f"/api/chat/sessions/{session_id}/image-analysis",
            files=files,
            data=data,
            headers=headers,
        )

    body = res.json()
    events = [e["event"] for e in body["execution_trace"]]

    # Required events for success
    assert "IMAGE_RECEIVED" in events
    assert "FILE_VALIDATED" in events
    assert "RBAC_CHECK" in events
    assert "VISION_MODEL_SELECTED" in events
    assert "VISION_ANALYSIS_STARTED" in events
    assert "OLLAMA_INFERENCE" in events
    assert "VISION_ANALYSIS_COMPLETED" in events
    assert "VISION_VERIFICATION_COMPLETED" in events
    assert "AUDIT" in events

    # Must NOT have failure events
    assert "VISION_ANALYSIS_FAILED" not in events
    assert "VISION_INFERENCE_FAILED" not in events


# ── 28. Execution Trace for OCR Fallback ──────────────────────────────────────

def test_28_execution_trace_ocr_fallback(client: TestClient):
    """Verify OCR fallback execution trace."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    session_res = client.post("/api/chat/sessions", json={"title": "Trace OCR Fallback"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    mock_500 = _make_mock_response(
        {"error": "prediction aborted"},
        status_code=500,
    )
    ocr_result = {
        "text": "Crude Oil\nGas Oil\nFurnace",
        "confidence": 0.9,
        "line_count": 3,
        "lines": [],
    }

    with patch("requests.post", return_value=mock_500):
        with patch(
            "backend.services.ocr.extract_text_from_image",
            return_value=ocr_result,
        ):
            files = {"file": ("ocr_trace.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
            data = {"prompt": "explain"}
            res = client.post(
                f"/api/chat/sessions/{session_id}/image-analysis",
                files=files,
                data=data,
                headers=headers,
            )

    body = res.json()
    events = [e["event"] for e in body["execution_trace"]]

    assert "VISION_OCR_FALLBACK" in events
    assert "VISION_ANALYSIS_FAILED" not in events  # OCR fallback is partial, not full fail


# ── 29. Non-Vision RAG Behavior Unchanged ─────────────────────────────────────

def test_29_non_vision_rag_unchanged(client: TestClient):
    """Existing text chat and RAG workflows must not be affected by vision changes."""
    token = _get_token(client, "engineer")
    headers = {"Authorization": f"Bearer {token}"}

    # Standard text message flow
    session_res = client.post("/api/chat/sessions", json={"title": "RAG Regression Session"}, headers=headers)
    assert session_res.status_code in [200, 201]
    session_id = session_res.json()["id"]

    msg_res = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"content": "What is the throughput capacity of the CDU?"},
        headers=headers,
    )
    assert msg_res.status_code == 201
    assert msg_res.json()["role"] == "user"

    # Session listing still works
    list_res = client.get("/api/chat/sessions", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS FOR VISION SERVICE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsVisionFailureResponse:
    """Unit tests for is_vision_failure_response()."""

    def test_http_500_is_failure(self):
        assert is_vision_failure_response(500, '{"error":"something"}') is True

    def test_http_200_with_valid_text_is_success(self):
        assert is_vision_failure_response(200, "This diagram shows a process flow") is False

    def test_http_200_with_empty_text_is_failure(self):
        assert is_vision_failure_response(200, "") is True
        assert is_vision_failure_response(200, "   ") is True

    def test_http_200_with_token_repeat_is_failure(self):
        assert is_vision_failure_response(200, "token repeat limit reached") is True

    def test_http_200_with_prediction_aborted_is_failure(self):
        assert is_vision_failure_response(200, "prediction aborted") is True

    def test_http_404_is_failure(self):
        assert is_vision_failure_response(404, "model not found") is True

    def test_http_200_with_json_error_envelope(self):
        assert is_vision_failure_response(200, '{"error":"something went wrong"}') is True


class TestBuildShortVisionPrompt:
    """Unit tests for the short vision prompt builder."""

    def test_prompt_contains_numbered_items(self):
        prompt = build_short_vision_prompt("explain this diagram")
        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt
        assert "4." in prompt
        assert "5." in prompt
        assert "6." in prompt

    def test_prompt_contains_user_query(self):
        prompt = build_short_vision_prompt("what is this process?")
        assert "what is this process?" in prompt

    def test_prompt_anti_hallucination_instructions(self):
        prompt = build_short_vision_prompt("explain")
        assert "Do not repeat" in prompt
        assert "Do not invent" in prompt


class TestVisionGenerationConfig:
    """Unit tests for vision generation configuration."""

    def test_vision_options_conservative(self):
        assert VISION_GENERATION_OPTIONS["temperature"] == 0.05
        assert VISION_GENERATION_OPTIONS["repeat_penalty"] == 1.1
        assert VISION_GENERATION_OPTIONS["num_predict"] in (512, 1024)

    def test_retry_options_stricter(self):
        assert VISION_RETRY_OPTIONS["repeat_penalty"] > VISION_GENERATION_OPTIONS["repeat_penalty"]
        assert VISION_RETRY_OPTIONS["num_predict"] < VISION_GENERATION_OPTIONS["num_predict"]

    def test_retry_options_lower_output_limit(self):
        assert VISION_RETRY_OPTIONS["num_predict"] == 256
