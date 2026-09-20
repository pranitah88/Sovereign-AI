"""
MRPL Sovereign AI Workbench — Modular Text-to-Speech (TTS) Service.
Strictly On-Premise / Local Inference — Zero Cloud Egress.
"""

from abc import ABC, abstractmethod
import io
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

from backend.services.network_seal import record_local_call
from backend.services.voice.speech_renderer import render_speech_text

logger = logging.getLogger(__name__)


def clean_text_for_speech(text: str) -> str:
    """
    Clean markdown, URLs, citation footnotes, and code fences for natural speech.
    Preserves technical equipment tags and engineering units.
    """
    if not text:
        return ""
    return render_speech_text(text)


class BaseTTSEngine(ABC):
    """Abstract interface for local text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str, language: str = "en") -> dict[str, Any]:
        """
        Synthesize text to speech audio bytes.

        Returns:
            {
                "audio_bytes": bytes,
                "content_type": "audio/wav",
                "duration_seconds": float,
                "engine": str,
                "status": "success" | "error",
                "error": str | None,
            }
        """
        pass

    @abstractmethod
    def synthesize_chunk(self, chunk: str, language: str = "en") -> dict[str, Any]:
        """Synthesize a single sentence/chunk for real-time streaming playback."""
        pass

    @abstractmethod
    def is_ready(self) -> tuple[bool, str]:
        """Return (is_ready, status_message)."""
        pass

    def warmup(self) -> bool:
        """Warm up engine weights or COM interfaces."""
        return True


class LocalSapiTTSEngine(BaseTTSEngine):
    """
    On-premise Windows SAPI TTS Engine using local speech synthesizer.
    Generates standard PCM WAV audio stream with zero external network access.
    """

    _lock = threading.Lock()

    def __init__(self, voice_rate: int = 165, voice_volume: float = 1.0):
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume

    def is_ready(self) -> tuple[bool, str]:
        ready = False
        msg = "No SAPI voices installed on system"
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            try:
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voices = voice.GetVoices()
                count = voices.Count
                if count > 0:
                    desc = str(voices.Item(0).GetDescription())
                    ready = True
                    msg = f"Windows SAPI Ready ({desc})"
                voices = None
                voice = None
            finally:
                pythoncom.CoUninitialize()
            return ready, msg
        except Exception as exc:
            return False, f"VOICE MODEL NOT INSTALLED: SAPI TTS error: {exc}"

    def synthesize(self, text: str, language: str = "en") -> dict[str, Any]:
        """Synthesize text into WAV bytes offline."""
        cleaned_speech = clean_text_for_speech(text)
        if not cleaned_speech:
            return {
                "audio_bytes": b"",
                "content_type": "audio/wav",
                "duration_seconds": 0.0,
                "engine": "Windows SAPI (local)",
                "status": "error",
                "error": "Empty text provided for synthesis",
            }

        record_local_call("local:sapi:tts_synthesize")

        with self._lock:
            temp_path = None
            try:
                import pythoncom
                import win32com.client

                pythoncom.CoInitialize()
                try:
                    voice = win32com.client.Dispatch("SAPI.SpVoice")
                    stream = win32com.client.Dispatch("SAPI.SpFileStream")

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        temp_path = tf.name

                    # 3 = SSFMCreateForWrite
                    stream.Open(temp_path, 3)
                    voice.AudioOutputStream = stream
                    
                    # Rate: -10 to +10 in SAPI. Map voice_rate (100-250) to -5 to +5
                    sapi_rate = max(min(int((self.voice_rate - 160) / 15), 10), -10)
                    voice.Rate = sapi_rate
                    voice.Volume = int(self.voice_volume * 100)

                    voice.Speak(cleaned_speech)
                    stream.Close()
                    stream = None
                    voice = None

                    with open(temp_path, "rb") as f:
                        wav_bytes = f.read()

                    # Approximate duration from WAV bytes (44.1kHz or 22kHz 16-bit mono/stereo)
                    duration_sec = max(len(wav_bytes) / 32000.0, 0.5)

                    return {
                        "audio_bytes": wav_bytes,
                        "content_type": "audio/wav",
                        "duration_seconds": round(duration_sec, 2),
                        "engine": "Windows SAPI (local)",
                        "status": "success",
                        "error": None,
                        "spoken_text": cleaned_speech,
                    }

                finally:
                    pythoncom.CoUninitialize()

            except Exception as exc:
                logger.error("TTS synthesis failed: %s", exc, exc_info=True)
                return {
                    "audio_bytes": b"",
                    "content_type": "audio/wav",
                    "duration_seconds": 0.0,
                    "engine": "Windows SAPI (local)",
                    "status": "error",
                    "error": f"TTS synthesis error: {str(exc)}",
                }
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    def synthesize_chunk(self, chunk: str, language: str = "en") -> dict[str, Any]:
        """Synthesize a single conversational chunk or sentence."""
        return self.synthesize(chunk, language=language)

    def warmup(self) -> bool:
        """Pre-warm SAPI COM interface to avoid first-call latency."""
        try:
            res = self.synthesize("Ready.", language="en")
            return res.get("status") == "success"
        except Exception as exc:
            logger.warning("Local SAPI TTS warmup non-fatal error: %s", exc)
            return False
