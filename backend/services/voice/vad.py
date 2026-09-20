"""
MRPL Sovereign AI Workbench — Voice Activity Detection (VAD) & Barge-In Service.
Strictly On-Premise / In-Memory Audio Analysis.
Detects speech onset, continuous speech, and end-of-speech silence thresholds.
"""

import io
import math
import struct
import wave
import numpy as np


class VoiceActivityDetector:
    """
    Local Voice Activity Detection based on normalized Root Mean Square (RMS) energy
    and adaptive silence framing.
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        silence_threshold_ms: int = 800,
        min_speech_duration_ms: int = 250,
        sample_rate: int = 16000,
    ):
        self.energy_threshold = energy_threshold
        self.silence_threshold_ms = silence_threshold_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.sample_rate = sample_rate

    @staticmethod
    def calculate_rms(audio_bytes: bytes) -> float:
        """Calculate normalized RMS energy from 16-bit PCM or WAV audio bytes."""
        if not audio_bytes:
            return 0.0

        raw_pcm = audio_bytes
        # If WAV header exists, strip header to read raw PCM frames
        if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 44:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    raw_pcm = wf.readframes(wf.getnframes())
            except Exception:
                raw_pcm = audio_bytes[44:]

        if len(raw_pcm) < 2:
            return 0.0

        # Unpack as int16 samples
        count = len(raw_pcm) // 2
        try:
            samples = struct.unpack(f"<{count}h", raw_pcm[: count * 2])
            if not samples:
                return 0.0
            sum_squares = sum(s * s for s in samples)
            rms = math.sqrt(sum_squares / count)
            # Normalize to 0.0 - 1.0 (int16 max = 32767)
            return rms / 32767.0
        except Exception:
            return 0.0

    def is_speech(self, audio_bytes: bytes, sensitivity: float | None = None) -> bool:
        """Return True if the audio chunk has energy exceeding the speech threshold."""
        thresh = sensitivity if sensitivity is not None else self.energy_threshold
        rms = self.calculate_rms(audio_bytes)
        return rms >= thresh

    def detect_speech_boundaries(
        self,
        audio_bytes: bytes,
        frame_duration_ms: int = 30,
    ) -> dict[str, any]:
        """
        Analyze audio frames to find speech start, speech end, and total speech duration.
        """
        raw_pcm = audio_bytes
        if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 44:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    raw_pcm = wf.readframes(wf.getnframes())
            except Exception:
                raw_pcm = audio_bytes[44:]

        bytes_per_frame = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2)
        if bytes_per_frame <= 0 or len(raw_pcm) < bytes_per_frame:
            return {
                "has_speech": False,
                "speech_duration_ms": 0,
                "silence_trailing_ms": 0,
                "speech_ended": False,
            }

        total_frames = len(raw_pcm) // bytes_per_frame
        speech_frames = 0
        consecutive_silence_frames = 0
        max_silence_frames = int(self.silence_threshold_ms / frame_duration_ms)
        speech_started = False
        speech_ended = False

        for i in range(total_frames):
            frame = raw_pcm[i * bytes_per_frame : (i + 1) * bytes_per_frame]
            frame_rms = self.calculate_rms(frame)

            if frame_rms >= self.energy_threshold:
                speech_frames += 1
                speech_started = True
                consecutive_silence_frames = 0
            else:
                if speech_started:
                    consecutive_silence_frames += 1
                    if consecutive_silence_frames >= max_silence_frames:
                        speech_ended = True

        speech_duration_ms = speech_frames * frame_duration_ms
        silence_trailing_ms = consecutive_silence_frames * frame_duration_ms

        has_speech = speech_duration_ms >= self.min_speech_duration_ms

        return {
            "has_speech": has_speech,
            "speech_started": speech_started,
            "speech_ended": speech_ended or (speech_started and silence_trailing_ms >= self.silence_threshold_ms),
            "speech_duration_ms": speech_duration_ms,
            "silence_trailing_ms": silence_trailing_ms,
        }
