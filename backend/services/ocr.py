"""
OCR service wrapper around PaddleOCR.

Provides clean interfaces for text extraction from images and PDF pages.
"""

import logging

logger = logging.getLogger(__name__)

# Lazy-loaded PaddleOCR instance.
_ocr_engine = None


def _get_ocr_engine():
    """Lazy-initialize PaddleOCR (heavy import, done once)."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        logger.info("PaddleOCR engine initialized")
    except ImportError as exc:
        logger.error("PaddleOCR is not installed: %s", exc)
        raise

    return _ocr_engine


def extract_text_from_image(image_path: str) -> dict:
    """
    Extract text from an image file using PaddleOCR.

    Returns:
        {
            "text": str,
            "confidence": float,  # average confidence (0-1)
            "line_count": int,
            "lines": list[{"text": str, "confidence": float}],
        }
    """
    ocr = _get_ocr_engine()
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        return {
            "text": "",
            "confidence": 0.0,
            "line_count": 0,
            "lines": [],
        }

    lines = []
    total_conf = 0.0

    for line in result[0]:
        text = line[1][0]
        conf = float(line[1][1])
        lines.append({"text": text, "confidence": conf})
        total_conf += conf

    full_text = "\n".join(l["text"] for l in lines)
    avg_confidence = total_conf / len(lines) if lines else 0.0

    logger.info(
        "OCR extracted %d lines from %s (avg confidence: %.2f)",
        len(lines),
        image_path,
        avg_confidence,
    )

    return {
        "text": full_text,
        "confidence": round(avg_confidence, 4),
        "line_count": len(lines),
        "lines": lines,
    }


def extract_text_from_pdf_page(page_image_path: str) -> dict:
    """
    Extract text from a rendered PDF page image.
    Same as extract_text_from_image — PDF pages should be
    pre-rendered to images by the caller (e.g. using PyMuPDF).
    """
    return extract_text_from_image(page_image_path)
