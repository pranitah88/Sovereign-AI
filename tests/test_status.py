"""
Tests for system monitoring and air-gap network interface status.
"""

from backend.services.system_monitor import (
    get_system_status,
    get_network_status,
    get_service_health,
)


def test_system_status_structure():
    status = get_system_status()
    assert "cpu" in status
    assert "memory" in status
    assert "disk" in status
    assert "gpu" in status

    # Verify memory structure
    mem = status["memory"]
    assert "total_gb" in mem
    assert "used_gb" in mem
    assert "usage_percent" in mem
    assert 0 <= mem["usage_percent"] <= 100


def test_network_status_structure():
    net = get_network_status()
    assert "interfaces" in net
    assert isinstance(net["interfaces"], dict)
    assert len(net["interfaces"]) > 0


def test_services_status_structure():
    services = get_service_health()
    assert "chromadb" in services
    assert "ollama" in services
    assert "n8n" in services
