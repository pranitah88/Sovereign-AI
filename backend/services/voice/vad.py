"""
MRPL Sovereign AI Workbench — Voice Activity Detection (VAD) & Barge-In Service.
Strictly On-Premise / In-Memory Audio Analysis.
Adaptive noise-floor tracking, SNR estimation, hysteresis, and digital high-pass filtering.
"""

import io
import math
import struct
from typing import Any
import wave
import numpy as np


def decode_audio_to_pcm(audio_bytes: bytes, target_sample_rate: int = 16000) -> tuple[np.ndarray, int, int]:
    """
    Decodes audio bytes (WebM Opus, OGG, WAV, MP3, or raw PCM) into a 1D float32 numpy array
    normalized to [-1.0, 1.0] at target_sample_rate (mono).
    
    Returns:
        (pcm_samples, sample_rate, channel_count)
    """
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32), target_sample_rate, 1

    # Fast-path for standard WAV
    if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 44:
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
            
            if sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sample_width == 1:
                samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)

            if sr != target_sample_rate and len(samples) > 0:
                try:
                    from scipy.signal import resample
                    target_len = int(len(samples) * (target_sample_rate / sr))
                    samples = resample(samples, target_len).astype(np.float32)
                except Exception:
                    pass
                sr = target_sample_rate

            return samples, sr, channels
        except Exception:
            pass

    # Universal container decoding via PyAV (WebM Opus, OGG, etc.)
    try:
        import av
        container = av.open(io.BytesIO(audio_bytes))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is not None:
            channels = audio_stream.channels or 1
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sample_rate)
            frame_chunks = []
            for frame in container.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    frame_chunks.append(resampled_frame.to_ndarray()[0])
            container.close()
            if frame_chunks:
                samples = np.concatenate(frame_chunks).astype(np.float32)
                return samples, target_sample_rate, channels
    except Exception:
        pass

    # Fallback raw 16-bit PCM
    count = len(audio_bytes) // 2
    if count > 0:
        try:
            samples = np.frombuffer(audio_bytes[: count * 2], dtype=np.int16).astype(np.float32) / 32768.0
            return samples, target_sample_rate, 1
        except Exception:
            pass

    return np.zeros(0, dtype=np.float32), target_sample_rate, 1


class VoiceActivityDetector:
    """
    Local Adaptive Voice Activity Detection based on:
    1. Digital high-pass filtering (attenuates fan/AC mechanical rumble < 80 Hz).
    2. Dynamic noise-floor tracking during non-speech intervals.
    3. Estimated Signal-to-Noise Ratio (SNR) in decibels.
    4. Dual-threshold hysteresis (start vs stop threshold) to prevent rapid flickering.
    5. Exposing real audio diagnostics (noise_floor_rms, speech_rms, snr_db, vad_confidence).
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        silence_threshold_ms: int = 750,
        min_speech_duration_ms: int = 250,
        sample_rate: int = 16000,
        adaptive: bool = True,
        min_energy_threshold: float = 0.012,
        snr_start_threshold_db: float = 6.0,
        snr_stop_threshold_db: float = 3.0,
        noise_floor_smoothing: float = 0.95,
        highpass_cutoff_hz: float = 80.0,
    ):
        self.energy_threshold = energy_threshold
        self.silence_threshold_ms = silence_threshold_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.sample_rate = sample_rate
        self.adaptive = adaptive
        self.min_energy_threshold = min_energy_threshold
        self.snr_start_threshold_db = snr_start_threshold_db
        self.snr_stop_threshold_db = snr_stop_threshold_db
        self.noise_floor_smoothing = noise_floor_smoothing
        self.highpass_cutoff_hz = highpass_cutoff_hz

        # Adaptive state tracking
        self.noise_floor_rms: float = 0.008
        self.speech_rms: float = 0.0
        self.current_rms: float = 0.0
        self.snr_db: float = 0.0
        self.vad_confidence: float = 0.0
        self.speech_active: bool = False

    @staticmethod
    def calculate_rms(audio_input: bytes | np.ndarray) -> float:
        """Calculate normalized RMS energy from audio bytes (WAV/WebM/PCM) or float32 array."""
        if isinstance(audio_input, np.ndarray):
            if len(audio_input) == 0:
                return 0.0
            return float(np.sqrt(np.mean(audio_input ** 2)))

        if not audio_input:
            return 0.0

        samples, _, _ = decode_audio_to_pcm(audio_input)
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    def apply_highpass_filter(self, audio_bytes: bytes, cutoff_hz: float | None = None) -> bytes:
        """
        Apply a digital single-pole IIR high-pass filter to remove low-frequency rumble
        (e.g., fan rumble, AC noise, handling noise) without attenuating speech formants.
        Cutoff is typically 70-100 Hz (default: self.highpass_cutoff_hz).
        """
        cutoff = cutoff_hz or self.highpass_cutoff_hz
        if not audio_bytes or cutoff <= 0:
            return audio_bytes

        raw_pcm = audio_bytes
        is_wav = False
        wav_params = None
        if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 44:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    wav_params = wf.getparams()
                    raw_pcm = wf.readframes(wf.getnframes())
                    is_wav = True
            except Exception:
                raw_pcm = audio_bytes[44:]

        count = len(raw_pcm) // 2
        if count < 2:
            return audio_bytes

        try:
            samples = np.frombuffer(raw_pcm[: count * 2], dtype=np.int16).astype(np.float32)
            try:
                from scipy.signal import butter, sosfilt
                sos = butter(2, cutoff, btype="highpass", fs=self.sample_rate, output="sos")
                filtered = sosfilt(sos, samples)
            except Exception:
                # Fallback 1st-order IIR High-Pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
                rc = 1.0 / (2.0 * math.pi * cutoff)
                dt = 1.0 / self.sample_rate
                alpha = rc / (rc + dt)
                filtered = np.zeros_like(samples)
                if len(samples) > 1:
                    filtered[0] = samples[0]
                    prev_y = filtered[0]
                    prev_x = samples[0]
                    for i in range(1, len(samples)):
                        curr_x = samples[i]
                        curr_y = alpha * (prev_y + curr_x - prev_x)
                        filtered[i] = curr_y
                        prev_y = curr_y
                        prev_x = curr_x

            out_samples = np.clip(filtered, -32768, 32767).astype(np.int16)
            out_bytes = out_samples.tobytes()

            if is_wav and wav_params:
                bio = io.BytesIO()
                with wave.open(bio, "wb") as out_wav:
                    out_wav.setparams(wav_params)
                    out_wav.writeframes(out_bytes)
                return bio.getvalue()

            return out_bytes
        except Exception:
            return audio_bytes

    def update_noise_floor(self, frame_rms: float, is_speech: bool = False, force: bool = False) -> float:
        """
        Dynamically estimate ambient noise floor during non-speech intervals.
        Uses exponential moving average with noise_floor_smoothing.
        Fast downward adaptation, smoothed upward adaptation with spike protection.
        """
        if frame_rms <= 0:
            return self.noise_floor_rms

        if frame_rms < self.noise_floor_rms:
            # Fast tracking downward when background becomes quieter
            self.noise_floor_rms = 0.85 * self.noise_floor_rms + 0.15 * frame_rms
        elif not is_speech or force:
            # Protect against sudden startup spikes or loud clicks: limit single-step increase
            max_step = self.noise_floor_rms * 1.35
            clipped_rms = min(frame_rms, max_step) if self.noise_floor_rms > 0.005 else frame_rms
            self.noise_floor_rms = (
                self.noise_floor_smoothing * self.noise_floor_rms
                + (1.0 - self.noise_floor_smoothing) * clipped_rms
            )
        self.noise_floor_rms = max(min(self.noise_floor_rms, 0.35), 1e-5)
        return self.noise_floor_rms

    def calculate_snr_db(self, frame_rms: float) -> float:
        """Calculate Signal-to-Noise Ratio in dB relative to current noise floor."""
        numerator = max(frame_rms, 1e-6)
        denominator = max(self.noise_floor_rms, 1e-6)
        return 20.0 * math.log10(numerator / denominator)

    def is_speech(self, audio_input: bytes | np.ndarray, sensitivity: float | None = None) -> bool:
        """
        Determine if the audio chunk contains active speech.
        In adaptive mode: evaluates SNR and dual-threshold hysteresis.
        In fixed mode or if sensitivity provided: evaluates threshold directly.
        """
        rms = self.calculate_rms(audio_input)
        self.current_rms = rms

        if sensitivity is not None:
            return rms >= sensitivity

        if not self.adaptive:
            return rms >= self.energy_threshold

        snr_db = self.calculate_snr_db(rms)
        self.snr_db = snr_db

        if self.speech_active:
            is_active = (
                (snr_db >= self.snr_stop_threshold_db and rms >= self.min_energy_threshold * 0.8)
                or (rms >= self.noise_floor_rms * 1.3 and rms >= self.min_energy_threshold)
            )
        else:
            is_active = (
                (rms >= self.min_energy_threshold)
                and (snr_db >= self.snr_start_threshold_db or rms >= self.noise_floor_rms * 2.0)
            )

        self.speech_active = is_active

        if is_active:
            self.speech_rms = (
                0.9 * self.speech_rms + 0.1 * rms if self.speech_rms > 0 else rms
            )
            self.vad_confidence = min(max((snr_db - self.snr_stop_threshold_db) / 12.0, 0.1), 1.0)
        else:
            self.update_noise_floor(rms, is_speech=False)
            self.vad_confidence = 0.0

        return is_active

    def detect_speech_boundaries(
        self,
        audio_bytes: bytes,
        frame_duration_ms: int = 30,
        is_preprocessed: bool = False,
    ) -> dict[str, Any]:
        """
        Analyze audio frames to find speech start, speech end, total duration,
        and real-time ambient noise / SNR metrics.
        """
        # Avoid duplicate filtering if frontend preprocessing already applied 80 Hz highpass
        is_webm = audio_bytes.startswith(b"\x1a\x45\xdf\xa3") or b"webm" in audio_bytes[:100].lower()
        if not is_preprocessed and not is_webm and self.highpass_cutoff_hz > 0:
            audio_bytes = self.apply_highpass_filter(audio_bytes)

        samples, sample_rate, channels = decode_audio_to_pcm(audio_bytes, target_sample_rate=self.sample_rate)
        total_samples = len(samples)
        if total_samples == 0:
            return {
                "has_speech": False,
                "speech_started": False,
                "speech_ended": False,
                "speech_duration_ms": 0,
                "silence_trailing_ms": 0,
                "noise_floor_rms": round(float(self.noise_floor_rms), 4),
                "speech_rms": round(float(self.speech_rms), 4),
                "current_rms": 0.0,
                "peak": 0.0,
                "input_rms": 0.0,
                "snr_db": round(float(self.snr_db), 2),
                "vad_confidence": 0.0,
                "capture_duration_ms": 0,
                "sample_rate": self.sample_rate,
                "channel_count": 1,
            }

        duration_ms = int(total_samples / self.sample_rate * 1000.0)
        input_rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples))) if total_samples > 0 else 0.0

        samples_per_frame = int(self.sample_rate * (frame_duration_ms / 1000.0))
        if samples_per_frame <= 0 or total_samples < samples_per_frame:
            has_sp = input_rms >= self.min_energy_threshold
            return {
                "has_speech": has_sp,
                "speech_started": has_sp,
                "speech_ended": has_sp,
                "speech_duration_ms": duration_ms if has_sp else 0,
                "silence_trailing_ms": 0 if has_sp else duration_ms,
                "noise_floor_rms": round(float(self.noise_floor_rms), 4),
                "speech_rms": round(input_rms if has_sp else 0.0, 4),
                "current_rms": round(input_rms, 4),
                "peak": round(peak, 4),
                "input_rms": round(input_rms, 4),
                "snr_db": round(float(self.snr_db), 2),
                "vad_confidence": 0.8 if has_sp else 0.0,
                "capture_duration_ms": duration_ms,
                "sample_rate": sample_rate,
                "channel_count": channels,
            }

        total_frames = total_samples // samples_per_frame
        speech_frames = 0
        consecutive_silence_frames = 0
        max_silence_frames = max(1, int(self.silence_threshold_ms / frame_duration_ms))
        speech_started = False
        speech_ended = False

        self.speech_active = False
        measured_speech_energies = []
        measured_snr_values = []

        calib_frames = min(3, total_frames)
        for c in range(calib_frames):
            c_frame = samples[c * samples_per_frame : (c + 1) * samples_per_frame]
            c_rms = float(np.sqrt(np.mean(c_frame ** 2)))
            if c_rms < self.energy_threshold:
                self.update_noise_floor(c_rms, is_speech=False)

        for i in range(total_frames):
            frame = samples[i * samples_per_frame : (i + 1) * samples_per_frame]
            frame_rms = float(np.sqrt(np.mean(frame ** 2)))
            self.current_rms = frame_rms

            snr_db = self.calculate_snr_db(frame_rms)
            self.snr_db = snr_db

            if self.speech_active:
                frame_is_speech = (
                    (snr_db >= self.snr_stop_threshold_db and frame_rms >= self.min_energy_threshold * 0.8)
                    or (frame_rms >= self.noise_floor_rms * 1.3 and frame_rms >= self.min_energy_threshold)
                )
            else:
                frame_is_speech = (
                    (frame_rms >= self.min_energy_threshold)
                    and (snr_db >= self.snr_start_threshold_db or frame_rms >= self.noise_floor_rms * 2.0)
                )

            self.speech_active = frame_is_speech

            if frame_is_speech:
                speech_frames += 1
                speech_started = True
                consecutive_silence_frames = 0
                measured_speech_energies.append(frame_rms)
                measured_snr_values.append(self.snr_db)
            else:
                self.update_noise_floor(frame_rms, is_speech=False)
                if speech_started:
                    consecutive_silence_frames += 1
                    if consecutive_silence_frames >= max_silence_frames:
                        speech_ended = True

        speech_duration_ms = speech_frames * frame_duration_ms
        silence_trailing_ms = consecutive_silence_frames * frame_duration_ms

        has_speech = speech_duration_ms >= self.min_speech_duration_ms
        avg_speech_rms = float(np.mean(measured_speech_energies)) if measured_speech_energies else 0.0
        avg_snr_db = float(np.mean(measured_snr_values)) if measured_snr_values else 0.0
        confidence = min(max((avg_snr_db - self.snr_stop_threshold_db) / 12.0, 0.1), 1.0) if has_speech else 0.0

        self.speech_rms = avg_speech_rms
        self.snr_db = avg_snr_db
        self.vad_confidence = confidence

        return {
            "has_speech": has_speech,
            "speech_started": speech_started,
            "speech_ended": speech_ended or (speech_started and silence_trailing_ms >= self.silence_threshold_ms),
            "speech_duration_ms": speech_duration_ms,
            "silence_trailing_ms": silence_trailing_ms,
            "noise_floor_rms": round(float(self.noise_floor_rms), 4),
            "speech_rms": round(float(avg_speech_rms), 4),
            "current_rms": round(float(self.current_rms), 4),
            "peak": round(float(peak), 4),
            "input_rms": round(float(input_rms), 4),
            "snr_db": round(float(avg_snr_db), 2),
            "vad_confidence": round(float(confidence), 2),
            "capture_duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "channel_count": channels,
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Return real development diagnostics for audio/VAD monitoring."""
        return {
            "noise_floor_rms": round(float(self.noise_floor_rms), 4),
            "speech_rms": round(float(self.speech_rms), 4),
            "current_rms": round(float(self.current_rms), 4),
            "peak": 0.0,
            "input_rms": round(float(self.current_rms), 4),
            "snr_db": round(float(self.snr_db), 2),
            "vad_confidence": round(float(self.vad_confidence), 2),
            "speech_active": bool(self.speech_active),
            "highpass_cutoff_hz": self.highpass_cutoff_hz,
            "adaptive_vad_enabled": self.adaptive,
            "capture_duration_ms": 0,
            "speech_duration_ms": 0,
            "sample_rate": self.sample_rate,
            "channel_count": 1,
        }
