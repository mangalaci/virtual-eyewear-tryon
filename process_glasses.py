"""
Pipeline:
1. Gemini: szarakat eltavolitja a lencsekbol + lencseket chroma-key zolddel (#00FF00) tolti
2. PIL: zold (lencse) + feher (hatter) pixelek atlatszova tetele
3. Menti: static/glasses/{id}.png
Igy barmelyik keretszinnel mukodik.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image
from google import genai
from google.genai import types

API_KEY = "AIzaSyAbRBGNHLD58EyDLATd4f4qBNw8xFMrbAk"
MODEL   = "gemini-3.1-flash-image-preview"

PROMPT = """Edit this eyeglasses product photo:

1. Remove any temple arms visible through the lenses.
2. Fill the lens areas with pure chroma-key green (#00FF00). Every pixel inside the lens openings must be exactly #00FF00.
3. Keep the frame, nose pads and bridge exactly as they are.
4. Keep the white background exactly as it is.

The result should show only the glasses frame on a white background, with bright green lenses."""

BASE_DIR  = Path(__file__).parent
RAW_DIR   = BASE_DIR / "static" / "glasses" / "raw"
OUT_DIR   = BASE_DIR / "static" / "glasses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["pepe", "reebok", "guess"]

client = genai.Client(api_key=API_KEY)

# Kuszobertekek
WHITE_THRESHOLD = 230   # hatter eltavolitasa
GREEN_THRESHOLD = 80    # chroma-key zold lencse eltavolitasa


def gemini_clean(image_bytes: bytes) -> bytes:
    """Gemini-vel tisztitja a lencseket es zolddel tolti."""
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise ValueError("Gemini nem adott vissza kepet")


def chroma_to_transparent(image_bytes: bytes) -> bytes:
    """
    Zold (lencse) es feher (hatter) pixeleket atlatszova teszi.
    Zold: G csatorna >> R es B  (chroma key)
    Feher: mindharom csatorna >= WHITE_THRESHOLD
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.uint8)

    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]

    green_mask = (
        (g.astype(int) - r.astype(int) > GREEN_THRESHOLD) &
        (g.astype(int) - b.astype(int) > GREEN_THRESHOLD)
    )
    white_mask = (r >= WHITE_THRESHOLD) & (g >= WHITE_THRESHOLD) & (b >= WHITE_THRESHOLD)

    data[..., 3] = np.where(green_mask | white_mask, 0, a)

    result = Image.fromarray(data, "RGBA")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def main():
    for prod_id in PRODUCTS:
        src = RAW_DIR / f"{prod_id}.jpg"
        if not src.exists():
            print(f"[{prod_id}] HIANYZIK: {src}")
            continue

        print(f"\n[{prod_id}]")

        # 1. Gemini: szar eltavolitasa + lencsek feherrel
        print("  1. Gemini feldolgozas...")
        raw_bytes = src.read_bytes()
        try:
            cleaned_bytes = gemini_clean(raw_bytes)
            print("     OK")
        except Exception as e:
            print(f"     HIBA: {e}")
            continue

        # Kozbulso eredmeny mentese
        gemini_out = RAW_DIR / f"{prod_id}_gemini.jpg"
        gemini_out.write_bytes(cleaned_bytes)
        print(f"     Mentve: {gemini_out.name}")

        # 2. Zold + feher -> atlatszo
        print("  2. Chroma-key + hatter eltavolitasa...")
        try:
            png_bytes = chroma_to_transparent(cleaned_bytes)
            print("     OK")
        except Exception as e:
            print(f"     HIBA: {e}")
            continue

        # 3. Mentes
        out = OUT_DIR / f"{prod_id}.png"
        out.write_bytes(png_bytes)
        size_kb = len(png_bytes) // 1024
        print(f"  3. Mentve: {out.name} ({size_kb} KB)")

    print("\nKesz.")


if __name__ == "__main__":
    main()
