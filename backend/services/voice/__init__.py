"""
MRPL Sovereign AI Workbench — Multilingual Voice Assistant Service Package.
Modular on-premise ASR, Language Detection, and TTS.
"""

from backend.services.voice.voice_service import get_voice_service, VoiceService

__all__ = ["get_voice_service", "VoiceService"]
