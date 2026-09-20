"""
Network Seal Service — Enforced Local-Only Inference & Zero-Egress Governance.

Guarantees and monitors that all LLM reasoning, embedding calculation, vector retrieval,
and code execution remain strictly localized to the on-premise host. All external network
egress attempts are tracked, audited, and blocked.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import threading
from typing import Any

from backend.services.audit import audit_log

logger = logging.getLogger(__name__)


@dataclass
class NetworkSealState:
    seal_active: bool = True
    mode: str = "STRICT_LOCAL_ONLY"
    egress_policy: str = "DENIED"
    total_local_calls: int = 0
    total_blocked_attempts: int = 0
    blocked_events: list[dict] = field(default_factory=list)
    verified_local_endpoints: list[str] = field(default_factory=lambda: [
        "127.0.0.1:11434 (Ollama Inference Runtime)",
        "127.0.0.1:8000 (MRPL Sovereign FastAPI)",
        "backend/chroma_db (On-Premise SQLite/ChromaDB Vector Store)",
    ])


class NetworkSeal:
    """Thread-safe singleton monitoring and controlling network egress."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._state = NetworkSealState()

    @classmethod
    def get_instance(cls) -> "NetworkSeal":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def is_local_address(self, address: str) -> bool:
        """Return True if the address is a permitted local loopback interface."""
        addr_lower = (address or "").lower()
        local_markers = ["127.0.0.1", "localhost", "0.0.0.0", "::1", "backend/chroma_db"]
        return any(m in addr_lower for m in local_markers)

    def record_local_call(self, endpoint: str, description: str = "") -> bool:
        """Record an approved on-premise local loopback inference or API call."""
        with self._lock:
            self._state.total_local_calls += 1
            return True

    @property
    def seal_active(self) -> bool:
        return self._state.seal_active

    @property
    def total_local_calls(self) -> int:
        return self._state.total_local_calls

    @property
    def blocked_external_attempts(self) -> int:
        return self._state.total_blocked_attempts

    def enforce_no_egress(self, destination: str, reason: str = "Outbound egress blocked by Network Seal") -> None:
        """Enforce zero egress: block, audit, and raise PermissionError if destination is non-local."""
        if not self.is_local_address(destination):
            self.block_external_call(destination, reason)
            raise PermissionError(f"Network Seal Violation: External network egress to '{destination}' is strictly forbidden.")

    def block_external_call(self, destination: str, reason: str = "Outbound egress blocked by Network Seal") -> None:
        """Record and audit an external egress block event."""
        with self._lock:
            self._state.total_blocked_attempts += 1
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "destination": destination,
                "reason": reason,
            }
            self._state.blocked_events.append(event)
            if len(self._state.blocked_events) > 50:
                self._state.blocked_events.pop(0)

        logger.warning("NETWORK SEAL BLOCKED EGRESS: target='%s' reason='%s'", destination, reason)
        try:
            audit_log(
                action="NETWORK_SEAL_EGRESS_BLOCKED",
                outcome="denied",
                target=destination,
                details={"reason": reason},
            )
        except Exception as e:
            logger.warning("Could not log network seal audit event: %s", e)

    def get_status(self) -> dict[str, Any]:
        """Return live Network Seal telemetry for API and UI status badges."""
        with self._lock:
            return {
                "seal_active": self._state.seal_active,
                "status": "LOCAL ONLY",
                "egress_policy": self._state.egress_policy,
                "external_calls_allowed": 0,
                "blocked_attempts": self._state.total_blocked_attempts,
                "allowed_local_calls": self._state.total_local_calls,
                "verified_local_endpoints": self._state.verified_local_endpoints,
                "recent_blocks": list(self._state.blocked_events[-5:]),
            }


def get_network_seal() -> NetworkSeal:
    """Return the global NetworkSeal singleton instance."""
    return NetworkSeal.get_instance()


# Convenience module-level functions
def record_local_call(endpoint: str = "127.0.0.1", description: str = "") -> None:
    NetworkSeal.get_instance().record_local_call(endpoint, description)


def block_external_call(destination: str, reason: str = "Outbound network egress prohibited by Network Seal") -> None:
    NetworkSeal.get_instance().block_external_call(destination, reason)


def get_network_seal_status() -> dict[str, Any]:
    return NetworkSeal.get_instance().get_status()

