"""
Tests for isolated code execution sandbox (Docker container-based).

Tests cover:
- Successful Python execution
- Runtime error handling
- Empty code rejection
- Unsupported language rejection
- Execution timeout
- Network isolation (no outbound connections)
- Host filesystem isolation (host paths inaccessible)
"""

import pytest

from backend.services.sandbox import check_sandbox_ready, execute_code


# ── Precondition: skip all tests if Docker sandbox is not ready ──────────

_sandbox_status = check_sandbox_ready()
_skip_reason = (
    None
    if _sandbox_status["ready"]
    else f"Docker sandbox not ready: {_sandbox_status.get('detail', _sandbox_status.get('reason'))}"
)

pytestmark = pytest.mark.skipif(
    not _sandbox_status["ready"],
    reason=_skip_reason or "Docker sandbox not ready",
)


# ── Core execution tests ────────────────────────────────────────────────


def test_sandbox_successful_python_execution():
    code = "print(2 + 2)\nprint('MRPL Sovereign AI')"
    res = execute_code(code, language="python", timeout_seconds=10)

    assert res["status"] == "success"
    assert res["exit_code"] == 0
    assert "4" in res["stdout"]
    assert "MRPL Sovereign AI" in res["stdout"]
    assert res["execution_time_ms"] > 0


def test_sandbox_failure_exit_code():
    code = "raise ValueError('Simulated failure in calculation')"
    res = execute_code(code, language="python", timeout_seconds=10)

    assert res["status"] == "failure"
    assert res["exit_code"] != 0
    assert "ValueError: Simulated failure in calculation" in res["stderr"]


def test_sandbox_empty_code():
    res = execute_code("", language="python")
    assert res["status"] == "error"
    assert "Empty code" in res["stderr"]


def test_sandbox_unsupported_language():
    res = execute_code("console.log('hi')", language="javascript")
    assert res["status"] == "error"
    assert "Unsupported language" in res["stderr"]


def test_sandbox_timeout():
    code = "import time; time.sleep(60)"
    res = execute_code(code, language="python", timeout_seconds=3)

    assert res["status"] == "timeout"
    assert "timed out" in res["stderr"]


# ── Isolation tests ─────────────────────────────────────────────────────


def test_sandbox_network_isolation():
    """Container must have no network access (network_mode='none')."""
    code = (
        "import socket\n"
        "try:\n"
        "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    s.settimeout(3)\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    print('CONNECTED')\n"
        "    s.close()\n"
        "except Exception as e:\n"
        "    print(f'BLOCKED: {e}')\n"
    )
    res = execute_code(code, language="python", timeout_seconds=10)

    assert res["status"] == "success"
    assert "BLOCKED" in res["stdout"]
    assert "CONNECTED" not in res["stdout"]


def test_sandbox_host_filesystem_isolation():
    """Container must not be able to access host/project paths."""
    code = (
        "import os\n"
        "host_paths = [\n"
        "    '/mnt/c/Users/PRANITA',\n"
        "    '/mnt/d/MRPL-Sovereign-AI',\n"
        "    '/mnt/c/Windows',\n"
        "    '/mnt/d',\n"
        "    '/mnt/c',\n"
        "]\n"
        "results = []\n"
        "for p in host_paths:\n"
        "    exists = os.path.exists(p)\n"
        "    results.append(f'{p}: {\"EXISTS\" if exists else \"INACCESSIBLE\"}')\n"
        "for r in results:\n"
        "    print(r)\n"
        "# Verify workspace contains only the script\n"
        "cwd_files = os.listdir('/sandbox')\n"
        "print(f'WORKSPACE_FILES: {cwd_files}')\n"
    )
    res = execute_code(code, language="python", timeout_seconds=10)

    assert res["status"] == "success"
    # All host mount paths must be inaccessible
    assert "EXISTS" not in res["stdout"]
    assert "INACCESSIBLE" in res["stdout"]
    # Workspace should contain only the script
    assert "script.py" in res["stdout"]
