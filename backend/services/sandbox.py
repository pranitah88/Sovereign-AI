"""
Code sandbox — Docker container-based isolated code execution.

Runs user code inside ephemeral Docker containers with hard resource
limits, network isolation, read-only root filesystem, and non-root
execution. No subprocess fallback — if Docker is unavailable, execution
is refused (fail closed).

Security posture:
- network_mode="none" — zero network access.
- read_only=True — immutable root filesystem.
- Non-root user (sandbox, uid 1000).
- mem_limit=256m, nano_cpus=1 CPU, pids_limit=64.
- No Docker socket mount, no privileged mode.
- Containers auto-removed after execution.
- /tmp mounted as tmpfs for Python runtime needs.
- Only a temporary workspace directory is bind-mounted (read-only).

Air-gapped operation:
- The sandbox image must be pre-built via explicit command:
    docker build -t mrpl-sandbox:latest -f backend/services/Dockerfile.sandbox backend/services/
- No image pull or build is attempted at runtime.
- If the image does not exist, a controlled error is returned.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

SANDBOX_IMAGE = os.getenv("MRPL_SANDBOX_IMAGE", "mrpl-sandbox:latest")
MAX_TIMEOUT = 60
DEFAULT_TIMEOUT = 30
MAX_OUTPUT_BYTES = 50_000
MEM_LIMIT = "256m"
NANO_CPUS = 1_000_000_000  # 1 CPU
PIDS_LIMIT = 64

# ── Singleton Docker client ─────────────────────────────────────────────

_docker_client = None
_docker_available = None


def _get_docker_client():
    """
    Return a singleton Docker client instance.

    Raises RuntimeError if Docker is not available.
    """
    global _docker_client, _docker_available

    if _docker_available is False:
        raise RuntimeError("Docker daemon is not available")

    if _docker_client is not None:
        return _docker_client

    try:
        import docker
        client = docker.from_env()
        client.ping()
        _docker_client = client
        _docker_available = True
        return client
    except Exception as exc:
        _docker_available = False
        raise RuntimeError(f"Docker daemon is not available: {exc}") from exc


def _check_sandbox_image(client) -> bool:
    """Check whether the sandbox Docker image exists locally."""
    try:
        client.images.get(SANDBOX_IMAGE)
        return True
    except Exception:
        return False


def check_sandbox_ready() -> dict:
    """
    Check if the Docker sandbox is ready for code execution.

    Returns a status dict with 'ready' boolean and details.
    Called during startup health checks and by the API layer.
    """
    try:
        client = _get_docker_client()
    except RuntimeError as exc:
        return {
            "ready": False,
            "reason": "docker_unavailable",
            "detail": str(exc),
        }

    if not _check_sandbox_image(client):
        return {
            "ready": False,
            "reason": "image_missing",
            "detail": (
                f"Sandbox image '{SANDBOX_IMAGE}' not found. "
                "Build it with: docker build -t mrpl-sandbox:latest "
                "-f backend/services/Dockerfile.sandbox backend/services/"
            ),
        }

    return {"ready": True}


def execute_code(
    code: str,
    language: str = "python",
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Execute code in an isolated Docker container.

    Returns:
        {
            "status": "success" | "failure" | "timeout" | "error",
            "stdout": str,
            "stderr": str,
            "exit_code": int | None,
            "execution_time_ms": int,
        }

    Fail-closed: if Docker is unavailable or the sandbox image is missing,
    returns status="error" without executing code on the host.
    """
    # ── Input validation ─────────────────────────────────────────────
    if not code.strip():
        return {
            "status": "error",
            "stdout": "",
            "stderr": "Empty code provided",
            "exit_code": None,
            "execution_time_ms": 0,
        }

    if language != "python":
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Unsupported language: {language}. Only 'python' is supported.",
            "exit_code": None,
            "execution_time_ms": 0,
        }

    timeout_seconds = min(max(timeout_seconds, 1), MAX_TIMEOUT)

    # ── Docker availability (fail closed) ────────────────────────────
    try:
        client = _get_docker_client()
    except RuntimeError as exc:
        logger.error("Sandbox unavailable — Docker daemon not reachable: %s", exc)
        return {
            "status": "error",
            "stdout": "",
            "stderr": "Code sandbox is unavailable: Docker daemon is not running.",
            "exit_code": None,
            "execution_time_ms": 0,
        }

    # ── Image existence check (no pull, no build) ────────────────────
    if not _check_sandbox_image(client):
        logger.error(
            "Sandbox image '%s' not found. "
            "Build with: docker build -t mrpl-sandbox:latest "
            "-f backend/services/Dockerfile.sandbox backend/services/",
            SANDBOX_IMAGE,
        )
        return {
            "status": "error",
            "stdout": "",
            "stderr": (
                f"Code sandbox is unavailable: image '{SANDBOX_IMAGE}' not found. "
                "An administrator must pre-build the sandbox image."
            ),
            "exit_code": None,
            "execution_time_ms": 0,
        }

    # ── Write code to temp directory ─────────────────────────────────
    container = None
    tmpdir = tempfile.mkdtemp(prefix="mrpl_sandbox_")

    try:
        script_path = Path(tmpdir) / "script.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.perf_counter()

        # ── Run container ────────────────────────────────────────────
        container = client.containers.run(
            image=SANDBOX_IMAGE,
            detach=True,
            # Mount only the temp directory with user code, read-only.
            volumes={
                tmpdir: {"bind": "/sandbox", "mode": "ro"},
            },
            # tmpfs for /tmp so Python can write .pyc, tempfiles, etc.
            tmpfs={"/tmp": "size=16m,noexec"},
            # Security: no network, read-only root, non-privileged.
            network_mode="none",
            read_only=True,
            privileged=False,
            # Resource limits.
            mem_limit=MEM_LIMIT,
            nano_cpus=NANO_CPUS,
            pids_limit=PIDS_LIMIT,
            # No Docker socket, no extra capabilities.
            security_opt=["no-new-privileges:true"],
            cap_drop=["ALL"],
            # User inside container (matches Dockerfile).
            user="1000:1000",
        )

        # ── Wait for completion with timeout ─────────────────────────
        try:
            result = container.wait(timeout=timeout_seconds)
            exit_code = result.get("StatusCode", -1)
        except Exception:
            # Timeout or wait failure — force stop.
            try:
                container.stop(timeout=1)
            except Exception:
                try:
                    container.kill()
                except Exception:
                    pass
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds",
                "exit_code": None,
                "execution_time_ms": elapsed_ms,
            }

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # ── Capture output ───────────────────────────────────────────
        try:
            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )[:MAX_OUTPUT_BYTES]
        except Exception:
            stdout = ""

        try:
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )[:MAX_OUTPUT_BYTES]
        except Exception:
            stderr = ""

        status = "success" if exit_code == 0 else "failure"

        return {
            "status": status,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "execution_time_ms": elapsed_ms,
            "sandbox_info": {
                "sandbox": "Docker",
                "network": "Disabled",
                "filesystem": "Read-only",
                "user": "Non-root",
                "exit_code": exit_code,
                "status": "Verified" if exit_code == 0 else "Failed",
            },
        }

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000) if "start" in dir() else 0
        logger.error("Sandbox execution error: %s", exc)
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Sandbox execution error: {exc}",
            "exit_code": None,
            "execution_time_ms": elapsed_ms,
        }

    finally:
        # ── Cleanup: always remove container and temp directory ───────
        if container is not None:
            try:
                container.remove(force=True)
            except Exception as exc:
                logger.warning("Failed to remove sandbox container: %s", exc)

        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
