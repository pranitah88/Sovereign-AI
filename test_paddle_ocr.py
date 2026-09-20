import io
import numpy as np
import pymupdf
from PIL import Image
from paddleocr import PaddleOCR


PDF_PATH = r"D:\MRPL-Sovereign-AI\merged_knowledge_base\02_Environment_Compliance\CFO for MRPL Refinery Complex (Consent No. AW-332309 Valid upto 30-06-2026) आकार 2.13 MB भाषा अंग्रेज़ी आरूप PDF.pdf"


print("=" * 70)
print("PADDLEOCR SINGLE-PDF TEST")
print("=" * 70)

print("\nLoading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("OCR engine loaded successfully.")

print("\nOpening PDF...")

doc = pymupdf.open(PDF_PATH)

print(f"PDF pages: {doc.page_count}")

page = doc.load_page(0)

print("\nRendering page 1 at 200 DPI...")

pix = page.get_pixmap(dpi=200)

img = Image.open(
    io.BytesIO(
        pix.tobytes("png")
    )
).convert("RGB")

print(f"PIL image size: {img.size}")

# IMPORTANT:
# PaddleOCR requires NumPy array or image path.
img_arr = np.array(img)

print(f"NumPy type: {type(img_arr)}")
print(f"NumPy shape: {img_arr.shape}")
print(f"NumPy dtype: {img_arr.dtype}")

print("\nRunning PaddleOCR...")

try:
    results = ocr.predict(img_arr)

    print("\nRESULT TYPE:")
    print(type(results))

    print("\nRESULT COUNT:")
    print(len(results))

    all_text = []

    for result_index, res in enumerate(results):

        print(f"\n--- Result {result_index} ---")

        try:
            rec_texts = res["rec_texts"]
        except Exception:
            rec_texts = getattr(res, "rec_texts", None)

        print("Recognized text lines:")
        print(rec_texts)

        if rec_texts:
            for text in rec_texts:
                if text:
                    all_text.append(str(text))

    final_text = "\n".join(all_text)

    print("\n" + "=" * 70)
    print("OCR OUTPUT")
    print("=" * 70)

    print(final_text[:5000])

    print("\n" + "=" * 70)
    print(f"TOTAL OCR CHARACTERS: {len(final_text)}")
    print(f"TOTAL OCR WORDS: {len(final_text.split())}")
    print("=" * 70)

except Exception as e:

    print("\n" + "=" * 70)
    print("OCR ERROR")
    print("=" * 70)

    print(type(e).__name__)
    print(str(e))

finally:
    doc.close()