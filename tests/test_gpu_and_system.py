"""
Tests for GPU acceleration, Ollama inference, runtime diagnostics,
CPU fallback, and Docker sandbox isolation.
"""

import inspect
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.services.system_monitor import (
    _get_gpu_info,
    get_security_status,
    get_system_status,
)


def test_gpu_detection_logic_structure():
    """Verify _get_gpu_info returns required diagnostic fields."""
    info = _get_gpu_info()

    assert "status" in info
    assert info["status"] in {"AVAILABLE", "UNAVAILABLE"}
    assert "gpu_available" in info
    assert isinstance(info["gpu_available"], bool)
    assert "gpu_name" in info
    assert "inference_backend" in info
    assert info["inference_backend"] == "ollama"
    assert "inference_device" in info
    assert info["inference_device"] in {"gpu", "cpu", "unknown"}
    assert "gpus" in info
    assert isinstance(info["gpus"], list)

    if info["gpu_available"]:
        assert info["status"] == "AVAILABLE"
        assert len(info["gpus"]) > 0
        assert info["gpu_name"] != "unknown"
        assert info["vram_total_mb"] is not None
        assert info["vram_total_mb"] > 0
        assert info["vram_used_mb"] is not None
        assert info["vram_free_mb"] is not None
        assert info["vram_free_mb"] >= 0


def test_system_status_diagnostics_contract():
    """
    Verify get_system_status() implements the diagnostics contract:
    {
      "gpu_available": bool,
      "gpu_name": str,
      "inference_backend": "ollama",
      "inference_device": "gpu" | "cpu" | "unknown"
    }
    """
    status = get_system_status()

    assert "gpu_available" in status
    assert isinstance(status["gpu_available"], bool)
    assert "gpu_name" in status
    assert isinstance(status["gpu_name"], str)
    assert "inference_backend" in status
    assert status["inference_backend"] == "ollama"
    assert "inference_device" in status
    assert status["inference_device"] in {"gpu", "cpu", "unknown"}

    # Preserves backwards-compatible resource objects
    assert "gpu" in status
    assert "cpu" in status
    assert "memory" in status
    assert "disk" in status


def test_cpu_fallback_when_gpu_unavailable():
    """
    When nvidia-smi fails or GPU is absent, verify graceful CPU fallback:
    gpu_available=False, status='UNAVAILABLE', inference_device='cpu' (if Ollama up).
    """
    mock_run = MagicMock()
    mock_run.returncode = 1
    mock_run.stdout = ""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": []}

    with patch("subprocess.run", return_value=mock_run):
        with patch("requests.get", return_value=mock_resp):
            info = _get_gpu_info()
            assert info["gpu_available"] is False
            assert info["status"] == "UNAVAILABLE"
            assert info["gpu_name"] == "unknown"
            assert info["inference_device"] == "cpu"
            assert info["vram_used_mb"] is None

            sys_status = get_system_status()
            assert sys_status["gpu_available"] is False
            assert sys_status["inference_device"] == "cpu"


def test_cpu_fallback_when_nvidia_smi_not_found():
    """When nvidia-smi binary does not exist, system cleanly falls back to CPU."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": []}

    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        with patch("requests.get", return_value=mock_resp):
            info = _get_gpu_info()
            assert info["gpu_available"] is False
            assert info["status"] == "UNAVAILABLE"
            assert info["reason"] == "nvidia-smi not found"
            assert info["inference_device"] == "cpu"


def test_security_status_reports_real_gpu_metrics():
    """Verify get_security_status() reports live GPU & inference device without hardcoding."""
    sec = get_security_status()

    assert "gpu_status" in sec
    assert sec["gpu_status"] in {"AVAILABLE", "UNAVAILABLE"}
    assert "inference_device" in sec
    assert sec["inference_device"] in {"OLLAMA GPU", "CPU FALLBACK", "UNKNOWN"}

    inner = sec["security_status"]
    assert "gpu_status" in inner
    assert "inference_device" in inner
    assert inner["verification_distinction"]["gpu_status"] == "VERIFIED BY APPLICATION"
    assert inner["verification_distinction"]["inference_device"] == "VERIFIED BY APPLICATION"


def test_no_hardcoded_gpu_pass_in_codebase():
    """Ensure no hardcoded 'GPU: PASS' or fake pass status is used."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files_to_check = [
        root / "backend" / "services" / "system_monitor.py",
        root / "backend" / "api" / "status.py",
        root / "frontend" / "src" / "pages" / "DashboardPage.jsx",
    ]

    for path in files_to_check:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            assert "GPU: PASS" not in content, f"Hardcoded 'GPU: PASS' found in {path}"
            assert '"GPU: PASS"' not in content, f"Hardcoded '\"GPU: PASS\"' found in {path}"


def test_docker_sandbox_strictly_has_no_gpu_access():
    """
    Verify the Docker sandbox runner does NOT provide GPU access:
    Must not contain --gpus all, device_requests for GPU, or privileged mode.
    """
    from backend.services import sandbox

    source = inspect.getsource(sandbox.execute_code)
    assert "--gpus all" not in source
    assert "device_requests" not in source
    assert 'privileged=True' not in source
    assert 'privileged=False' in source
    assert 'network_mode="none"' in source
    assert 'read_only=True' in source


def test_live_ollama_gemma3_gpu_inference():
    """Verify live inference with Gemma 3 4B on local Ollama."""
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": "Respond with the word SUCCESS only.",
                "stream": False,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            pytest.skip("Ollama or gemma3:4b not available on this host")
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0

        # Check /api/ps to confirm VRAM offloading
        ps = requests.get("http://127.0.0.1:11434/api/ps", timeout=5).json()
        active_models = [m for m in ps.get("models", []) if "gemma3" in m.get("name", "")]
        if active_models:
            # Active Gemma model has VRAM allocation
            assert active_models[0].get("size_vram", 0) > 0

    except requests.RequestException:
        pytest.skip("Ollama service not running")


def test_live_ollama_qwen_coder_gpu_inference():
    """Verify live inference with Qwen 2.5 Coder 3B on local Ollama."""
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "qwen2.5-coder:3b",
                "prompt": "def add(a, b): return a + b",
                "stream": False,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            pytest.skip("Ollama or qwen2.5-coder:3b not available on this host")
        data = resp.json()
        assert "response" in data
        assert len(data["response"]) > 0

        # Check /api/ps to confirm VRAM offloading
        ps = requests.get("http://127.0.0.1:11434/api/ps", timeout=5).json()
        active_models = [m for m in ps.get("models", []) if "qwen" in m.get("name", "")]
        if active_models:
            # Active Qwen model has VRAM allocation
            assert active_models[0].get("size_vram", 0) > 0

    except requests.RequestException:
        pytest.skip("Ollama service not running")
