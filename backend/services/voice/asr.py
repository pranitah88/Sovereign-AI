"""
MRPL Sovereign AI Workbench — Modular Automated Speech Recognition (ASR).
Strictly On-Premise / Local Inference — Zero Cloud Egress.
"""

from abc import ABC, abstractmethod
import io
import logging
from pathlib import Path
import re
from typing import Any

from backend.services.network_seal import record_local_call

logger = logging.getLogger(__name__)

# Known industrial technical terms and equipment tags regex for preservation
INDUSTRIAL_ACRONYMS = [
    "CDU", "VDU", "HCU", "PFCCU", "FCCU", "DHDT", "CCR", "HGU", "SRU", "DCU",
    "PPU", "P&ID", "PID", "HSE", "MRPL", "OMPL", "LPG", "MS", "HSD", "ATF",
    "FO", "LOBS", "ISOM", "DHT", "NHT", "SWS", "ARU", "ETP",
]


def load_asr_gate_config() -> dict[str, Any]:
    """Load configurable ASR quality gate thresholds from config/voice.yaml."""
    cfg_path = Path("config/voice.yaml")
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                gate = data.get("voice", {}).get("asr_quality_gate", {})
                if gate:
                    return gate
        except Exception:
            pass
    return {
        "enabled": True,
        "min_token_prob": 0.55,
        "min_language_prob": 0.35,
        "max_repetition_ngram_count": 3,
        "min_vocab_diversity": 0.40,
        "supported_languages": ["en", "hi", "mr", "kn"],
    }


def validate_asr_quality(
    text: str,
    confidence: float = 1.0,
    language: str | None = None,
    language_prob: float = 1.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministic ASR quality validation gate.
    Evaluates:
      - Average token probability
      - Language probability
      - Detected language against supported languages (en, hi, mr, kn)
      - Repeated n-grams
      - Vocabulary diversity
      - Abnormal character / token sequences (e.g. Turkish / foreign characters)
      - Pathological repetition
      - Transcript length anomalies & noise hallucinations

    Rejects pathological or low-confidence speech with low_confidence=True
    instead of blindly sending bad transcripts downstream to RAG or LLM.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "valid": False,
            "status": "empty",
            "low_confidence": True,
            "reason": "Empty transcript",
            "spoken_prompt": "Sorry, I didn't catch that. Could you repeat it?",
        }

    gate_cfg = config or load_asr_gate_config()
    min_token_prob = float(gate_cfg.get("min_token_prob", 0.55))
    min_lang_prob = float(gate_cfg.get("min_language_prob", 0.35))
    max_ngram_reps = int(gate_cfg.get("max_repetition_ngram_count", 3))
    min_vocab_div = float(gate_cfg.get("min_vocab_diversity", 0.40))
    supported_langs = gate_cfg.get("supported_languages", ["en", "hi", "mr", "kn"])

    lower_text = clean_text.lower().rstrip(".,!?")

    # 1. Check for abnormal / non-supported foreign characters (e.g. Turkish ı, ş, ğ, Cyrillic, Chinese, etc.)
    # Indian languages use Devanagari, Kannada script, or standard Latin English
    foreign_chars = re.findall(r"[\u0131\u011f\u015f\u015e\u0130\u011e\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]", clean_text)
    if foreign_chars:
        return {
            "valid": False,
            "status": "low_confidence",
            "low_confidence": True,
            "reason": f"Abnormal/unsupported foreign characters detected: '{''.join(set(foreign_chars))}'",
            "low_confidence_reason": "foreign_characters",
            "spoken_prompt": "Sorry, I didn't catch that. Could you repeat it?",
        }

    # Common foreign / pathological phrases that get misclassified by Whisper
    pathological_foreign_patterns = [
        r"\bbu\s+arada\b",
        r"\byusuzlar\b",
        r"\byusuzlas\b",
    ]
    if any(re.search(p, lower_text) for p in pathological_foreign_patterns):
        return {
            "valid": False,
            "status": "low_confidence",
            "low_confidence": True,
            "reason": "Pathological foreign speech pattern detected",
            "low_confidence_reason": "foreign_characters",
            "spoken_prompt": "Sorry, I didn't catch that. Could you repeat it?",
        }

    # 2. Check supported language & language probability without falsely rejecting valid speech:
    # On short audio snippets (< 3s), Whisper-tiny's whole-audio language classifier frequently jitters
    # (e.g. mislabels English as 'nn', 'cy', 'la' or returns low language_prob).
    # If the text is composed of supported scripts (Latin, Devanagari, Kannada), downstream
    # language detection in multilingual.py accurately classifies it.
    is_supported_script = bool(re.search(r"[a-zA-Z0-9\u0900-\u097F\u0C80-\u0CFF]", clean_text))
    if language:
        clean_lang = language.lower().strip()
        if clean_lang not in ("auto", "unknown", "detect") and clean_lang not in supported_langs:
            if not is_supported_script or confidence < min_token_prob:
                return {
                    "valid": False,
                    "status": "low_confidence",
                    "low_confidence": True,
                    "reason": f"Unsupported detected language: '{clean_lang}' (supported: {supported_langs})",
                    "low_confidence_reason": "unsupported_language",
                    "spoken_prompt": "Sorry, I didn't catch that. Could you repeat it?",
                }

    # If language probability is low AND acoustic token confidence is also below threshold, reject.
    # Otherwise, if acoustic token confidence is solid, accept and let downstream multilingual handle it.
    if language_prob < min_lang_prob and confidence < min_token_prob:
        return {
            "valid": False,
            "status": "low_confidence",
            "low_confidence": True,
            "reason": f"Language confidence too low: {language_prob:.2f} < {min_lang_prob}",
            "low_confidence_reason": "low_token_confidence",
            "spoken_prompt": "Sorry, I didn't catch that. Could you repeat it?",
        }

    # 3. Common Whisper noise/silence hallucination phrases
    hallucination_phrases = [
        "thank you for watching",
        "thanks for watching",
        "subtitles by",
        "please subscribe",
        "like and subscribe",
        "bye bye",
        "you",
    ]
    if lower_text in hallucination_phrases:
        return {
            "valid": False,
            "status": "repetition_hallucination",
            "low_confidence": True,
            "reason": f"Noise hallucination phrase detected: '{lower_text}'",
            "low_confidence_reason": "noise_hallucination",
            "spoken_prompt": "Sorry, I didn't catch that. Could you say it again?",
        }

    words = re.findall(r"\b\w+\b", lower_text)
    if len(words) >= 3:
        # 4. Single word consecutive repetition (>= 3 in a row, e.g. "of of of" or "bit of bit of")
        consecutive_repeat_count = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                consecutive_repeat_count += 1
                if consecutive_repeat_count >= 3:
                    return {
                        "valid": False,
                        "status": "repetition_hallucination",
                        "low_confidence": True,
                        "reason": f"Consecutive repeated word: '{words[i]}'",
                        "low_confidence_reason": "repetition_detected",
                        "spoken_prompt": "Sorry, I didn't catch that. Could you say it again?",
                    }
            else:
                consecutive_repeat_count = 1

        # 5. Repeated n-grams (2-word to 5-word phrases appearing >= max_ngram_reps times)
        from collections import Counter
        for n in range(2, min(6, len(words) // 2 + 1)):
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)
            for ngram, count in counts.items():
                if count >= max_ngram_reps:
                    phrase = " ".join(ngram)
                    return {
                        "valid": False,
                        "status": "repetition_hallucination",
                        "low_confidence": True,
                        "reason": f"Pathological n-gram repetition ({count}x): '{phrase}'",
                        "low_confidence_reason": "repetition_detected",
                        "spoken_prompt": "Sorry, I didn't catch that. Could you say it again?",
                    }

        # 6. Low vocabulary diversity for runaway transcripts (>= 6 words with unique ratio < min_vocab_div)
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < min_vocab_div:
                return {
                    "valid": False,
                    "status": "repetition_hallucination",
                    "low_confidence": True,
                    "reason": f"Low vocabulary diversity ({unique_ratio:.2f} < {min_vocab_div})",
                    "low_confidence_reason": "repetition_detected",
                    "spoken_prompt": "Sorry, I didn't catch that. Could you say it again?",
                }

    # 7. Check acoustic token probability
    if confidence < min_token_prob:
        return {
            "valid": False,
            "status": "low_confidence",
            "low_confidence": True,
            "reason": f"Token acoustic confidence too low: {confidence:.2f} < {min_token_prob}",
            "low_confidence_reason": "low_token_confidence",
            "spoken_prompt": "Sorry, I didn't catch that. Could you repeat it?",
        }

    return {"valid": True, "status": "success", "low_confidence": False, "reason": "Quality verified", "low_confidence_reason": None}



class BaseASREngine(ABC):
    """Abstract interface for local speech-to-text engines."""

    @abstractmethod
    def transcribe(
        self,
        audio_data: bytes | io.BytesIO | str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio input to text.

        Returns:
            {
                "text": str,
                "language": str,
                "confidence": float,
                "status": "success" | "low_confidence" | "empty" | "error",
                "duration_seconds": float,
                "engine": str,
                "details": dict,
            }
        """
        pass

    @abstractmethod
    def is_ready(self) -> tuple[bool, str]:
        """Return (is_ready, status_message)."""
        pass

    def warmup(self) -> bool:
        """Pre-warm model weights. Default no-op for generic engines."""
        return True

    def transcribe_partial(
        self,
        audio_data: bytes | io.BytesIO | str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Fast low-latency partial transcription of live audio slices."""
        return self.transcribe(audio_data, language=language)


class LocalWhisperASR(BaseASREngine):
    """
    On-premise faster-whisper engine using locally cached weights.
    Runs on CPU/GPU with int8 quantization. Fully air-gapped with local_files_only=True.
    """

    def __init__(
        self,
        model_name: str = "tiny",
        model_path: str = "models/whisper",
        device: str = "cpu",
        compute_type: str = "int8",
        confidence_threshold: float = 0.55,
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.device = device
        self.compute_type = compute_type
        self.confidence_threshold = confidence_threshold
        self.local_files_only = local_files_only
        self._model = None
        self._initialized = False
        self._init_error = None

    def _load_model(self) -> None:
        """Initialize the local Whisper model lazily."""
        if self._initialized:
            return

        # Check if local model directory exists
        if not self.model_path.exists():
            self._init_error = f"VOICE MODEL NOT INSTALLED: Directory '{self.model_path}' not found."
            logger.error(self._init_error)
            return

        try:
            from faster_whisper import WhisperModel

            record_local_call(f"local:faster-whisper:{self.model_name}")
            logger.info(
                "Loading local ASR model '%s' from '%s' (device=%s, compute=%s, offline=%s)...",
                self.model_name,
                self.model_path,
                self.device,
                self.compute_type,
                self.local_files_only,
            )
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_path),
                local_files_only=self.local_files_only,
            )
            self._initialized = True
            logger.info("Local Whisper ASR initialized successfully.")
        except Exception as exc:
            self._init_error = f"VOICE MODEL NOT INSTALLED: Failed to load local model: {exc}"
            logger.error("Local Whisper ASR initialization error: %s", exc)

    def is_ready(self) -> tuple[bool, str]:
        self._load_model()
        if self._model is not None:
            return True, f"Local Whisper ({self.model_name}) Ready"
        return False, self._init_error or "VOICE MODEL NOT INSTALLED"

    def transcribe(
        self,
        audio_data: bytes | io.BytesIO | str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio strictly on-premise without external network calls."""
        self._load_model()
        if self._model is None:
            return {
                "text": "",
                "language": language or "unknown",
                "confidence": 0.0,
                "status": "error",
                "duration_seconds": 0.0,
                "engine": f"faster-whisper ({self.model_name})",
                "error": self._init_error or "VOICE MODEL NOT INSTALLED",
            }

        # Convert bytes to file-like BytesIO
        if isinstance(audio_data, bytes):
            if len(audio_data) < 500:
                return {
                    "raw_text": "",
                    "normalized_text": "",
                    "is_normalized": False,
                    "text": "",
                    "language": language or "unknown",
                    "confidence": 0.0,
                    "status": "empty",
                    "duration_seconds": 0.0,
                    "engine": f"faster-whisper ({self.model_name})",
                    "error": "Empty or insufficient audio payload received",
                }

            # WebM container hygiene: Ensure standard EBML header (0x1A 0x45 0xDF 0xA3)
            # If stray audio chunks were prepended due to continuous recording slicing, find the true EBML start
            ebml_magic = b"\x1a\x45\xdf\xa3"
            magic_idx = audio_data.find(ebml_magic)
            if magic_idx > 0:
                logger.info("Auto-repaired WebM container: stripped %d leading bytes to EBML header", magic_idx)
                audio_data = audio_data[magic_idx:]

            audio_stream = io.BytesIO(audio_data)
        elif isinstance(audio_data, io.BytesIO):
            audio_stream = audio_data
        elif isinstance(audio_data, str):
            audio_stream = audio_data
        else:
            return {
                "text": "",
                "language": language or "unknown",
                "confidence": 0.0,
                "status": "error",
                "duration_seconds": 0.0,
                "engine": f"faster-whisper ({self.model_name})",
                "error": f"Invalid audio input type: {type(audio_data)}",
            }

        record_local_call(f"local:faster-whisper:transcribe")

        try:
            # Map user language hint if given (e.g. 'auto' -> None)
            lang_param = language if language and language.lower() not in ("auto", "auto detect", "detect") else None
            
            segments, info = self._model.transcribe(
                audio_stream,
                language=lang_param,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                hallucination_silence_threshold=0.5,
            )

            # Collect segments and compute token-level confidence
            collected_text = []
            segment_confidences = []
            total_duration = getattr(info, "duration", 0.0) or 0.0

            for seg in segments:
                text_part = seg.text.strip()
                if text_part:
                    collected_text.append(text_part)
                # Compute approximate segment confidence from avg_logprob
                # avg_logprob is typically negative (-0.1 to -2.0)
                # probability ~= exp(avg_logprob)
                import math
                prob = math.exp(seg.avg_logprob) if seg.avg_logprob is not None else 0.8
                segment_confidences.append(min(max(prob, 0.0), 1.0))

            full_text = " ".join(collected_text).strip()
            detected_lang = getattr(info, "language", language or "en")
            
            # Pure token-level acoustic confidence (separate from language detection probability)
            avg_token_prob = (
                sum(segment_confidences) / len(segment_confidences)
                if segment_confidences
                else 0.8
            )
            overall_confidence = round(min(max(avg_token_prob, 0.0), 1.0), 2)
            lang_prob = getattr(info, "language_probability", 1.0) or 1.0

            # Deterministic normalization (separate raw transcribed speech from normalization)
            normalized_text, is_normalized = self.deterministic_normalize(full_text)

            if not normalized_text and not full_text:
                return {
                    "raw_text": "",
                    "normalized_text": "",
                    "is_normalized": False,
                    "text": "",
                    "language": detected_lang,
                    "confidence": overall_confidence,
                    "status": "empty",
                    "duration_seconds": total_duration,
                    "engine": f"faster-whisper ({self.model_name})",
                }

            # Check deterministic quality (detect pathological repetition, runaway repeats, noise hallucinations)
            quality = validate_asr_quality(
                normalized_text or full_text,
                confidence=overall_confidence,
                language=detected_lang,
                language_prob=lang_prob,
            )
            if not quality["valid"]:
                status = quality["status"]
                effective_confidence = 0.0
                low_conf_reason = quality.get("low_confidence_reason") or quality.get("reason")
            else:
                status = "success" if overall_confidence >= self.confidence_threshold else "low_confidence"
                effective_confidence = overall_confidence
                low_conf_reason = f"Token acoustic confidence too low: {overall_confidence:.2f} < {self.confidence_threshold:.2f}" if status == "low_confidence" else None

            is_low_confidence = (status == "low_confidence") or not quality["valid"]

            return {
                "raw_transcript": full_text,
                "raw_text": full_text,
                "normalized_query": normalized_text,
                "normalized_text": normalized_text,
                "current_user_query": normalized_text or full_text,
                "is_normalized": is_normalized,
                "text": normalized_text or full_text,
                "language": detected_lang,
                "confidence": effective_confidence,
                "status": status,
                "low_confidence": is_low_confidence,
                "low_confidence_reason": low_conf_reason,
                "duration_seconds": total_duration,
                "engine": f"faster-whisper ({self.model_name})",
                "quality": quality,
                "details": {
                    "language_probability": round(lang_prob, 2),
                    "segment_count": len(segment_confidences),
                    "model": self.model_name,
                    "raw_text": full_text,
                    "is_normalized": is_normalized,
                    "quality_reason": quality.get("reason"),
                    "low_confidence_reason": low_conf_reason,
                    "raw_confidence": overall_confidence,
                    "confidence_threshold": self.confidence_threshold,
                },
            }

        except Exception as exc:
            logger.error("ASR transcription failed: %s", exc, exc_info=True)
            return {
                "raw_transcript": "",
                "raw_text": "",
                "normalized_query": "",
                "normalized_text": "",
                "current_user_query": "",
                "is_normalized": False,
                "text": "",
                "language": language or "unknown",
                "confidence": 0.0,
                "status": "error",
                "low_confidence": True,
                "duration_seconds": 0.0,
                "engine": f"faster-whisper ({self.model_name})",
                "error": f"Transcription error: {str(exc)}",
            }

    @staticmethod
    def deterministic_normalize(text: str) -> tuple[str, bool]:
        """
        Deterministic normalization for recognized equipment tags and industrial acronyms.
        Strictly formatting normalization only — no LLM guessing, no fuzzy matching.
        Returns: (normalized_text, is_normalized)
        """
        if not text:
            return text, False
        from backend.services.voice.domain_vocabulary import normalize_asr_transcript
        return normalize_asr_transcript(text)


    @staticmethod
    def _preserve_technical_terms(text: str) -> str:
        norm, _ = LocalWhisperASR.deterministic_normalize(text)
        return norm

    def transcribe_partial(
        self,
        audio_data: bytes | io.BytesIO | str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Fast low-latency partial transcription of live audio slices.
        Uses beam_size=1 (greedy) without heavy VAD filtering to return interim text rapidly.
        """
        self._load_model()
        if self._model is None:
            return {"raw_text": "", "normalized_text": "", "is_normalized": False, "text": "", "status": "error", "error": self._init_error or "VOICE MODEL NOT INSTALLED"}

        if isinstance(audio_data, bytes):
            if len(audio_data) < 1600:  # < 0.05s of audio
                return {"raw_text": "", "normalized_text": "", "is_normalized": False, "text": "", "status": "empty"}

            ebml_magic = b"\x1a\x45\xdf\xa3"
            magic_idx = audio_data.find(ebml_magic)
            if magic_idx > 0:
                audio_data = audio_data[magic_idx:]

            audio_stream = io.BytesIO(audio_data)
        elif isinstance(audio_data, io.BytesIO):
            audio_stream = audio_data
        else:
            audio_stream = audio_data

        try:
            lang_param = language if language and language.lower() not in ("auto", "auto detect", "detect") else None
            segments, info = self._model.transcribe(
                audio_stream,
                language=lang_param,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                hallucination_silence_threshold=0.5,
            )
            partial_words = [seg.text.strip() for seg in segments if seg.text.strip()]
            partial_text = " ".join(partial_words).strip()
            normalized_partial, is_norm = self.deterministic_normalize(partial_text)
            detected_lang = getattr(info, "language", language or "en")

            # Deterministic quality validation on interim speech to drop repetition and noise hallucinations
            if normalized_partial or partial_text:
                quality = validate_asr_quality(normalized_partial or partial_text, confidence=1.0)
                if not quality["valid"]:
                    logger.warning("Interim ASR rejected by quality validator: %s", quality.get("reason"))
                    return {
                        "raw_text": "",
                        "normalized_text": "",
                        "is_normalized": False,
                        "text": "",
                        "language": detected_lang,
                        "status": quality["status"],
                        "quality": quality,
                        "is_final": False,
                    }

            return {
                "raw_text": partial_text,
                "normalized_text": normalized_partial,
                "is_normalized": is_norm,
                "text": normalized_partial,
                "language": detected_lang,
                "status": "partial",
                "is_final": False,
            }
        except Exception as exc:
            return {"raw_text": "", "normalized_text": "", "is_normalized": False, "text": "", "status": "error", "error": str(exc)}

    def warmup(self) -> bool:
        """Pre-warm local model into memory with tiny synthetic frame."""
        self._load_model()
        if self._model is None:
            return False
        try:
            blank = io.BytesIO(b"\x00" * 3200)
            self._model.transcribe(blank, beam_size=1)
            logger.info("Local Whisper ASR warmed up successfully.")
            return True
        except Exception as exc:
            logger.warning("Warmup run encountered non-fatal note: %s", exc)
            return True
