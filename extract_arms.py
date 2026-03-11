"""
Gemini-vel kiszedi a szarat az alt2 kepekbol.
Pipeline:
1. Gemini: csak a szar marad, feher hatterre, vizszintesre egyenesitve
2. PIL: feher -> atlatszo PNG
3. Menti: static/glasses/processed/{id}_arm.png
"""

import io
import numpy as np
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types

API_KEY = "AIzaSyAbRBGNHLD58EyDLATd4f4qBNw8xFMrbAk"
MODEL   = "gemini-3.1-flash-image-preview"

PROMPT = """From this eyeglasses image, extract ONLY the temple arm — the long thin horizontal stick that goes from the hinge to the ear tip.

Very important rules:
- The temple arm is the thin elongated bar, like a stick or rod
- Do NOT include: the front frame, the lens rims, the bridge, nose pads, or any curved frame parts
- Do NOT include the side edge/profile of the front frame — only the arm stick itself
- The result must be a single thin horizontal bar, like a pen or a stick
- Straighten it so it is perfectly horizontal
- Place it on a pure white (#FFFFFF) background
- No shadows, no reflections"""

BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "static" / "glasses" / "raw"
PROCESSED_DIR = BASE_DIR / "static" / "glasses" / "processed"
WHITE_THRESH  = 230

PRODUCTS = ["pepe", "reebok"]

client = genai.Client(api_key=API_KEY)


def gemini_extract_arm(image_bytes: bytes) -> bytes:
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


def white_to_transparent(image_bytes: bytes) -> bytes:
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    white_mask  = (r >= WHITE_THRESH) & (g >= WHITE_THRESH) & (b >= WHITE_THRESH)
    data[..., 3] = np.where(white_mask, 0, a)
    result = Image.fromarray(data, "RGBA")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def main():
    for prod_id in PRODUCTS:
        src = RAW_DIR / f"{prod_id}_alt2.jpg"
        if not src.exists():
            print(f"[{prod_id}] HIANYZIK: {src}")
            continue

        print(f"\n[{prod_id}]")

        print("  1. Gemini: szar kiszedese...")
        try:
            arm_bytes = gemini_extract_arm(src.read_bytes())
            print("     OK")
        except Exception as e:
            print(f"     HIBA: {e}")
            continue

        # Kozbulso mentes
        mid = RAW_DIR / f"{prod_id}_arm_raw.jpg"
        mid.write_bytes(arm_bytes)
        print(f"     Mentve: {mid.name}")

        print("  2. Feher -> atlatszo...")
        png_bytes = white_to_transparent(arm_bytes)
        print("     OK")

        out = PROCESSED_DIR / f"{prod_id}_arm.png"
        out.write_bytes(png_bytes)
        print(f"  3. Mentve: {out.name} ({len(png_bytes)//1024} KB)")

    print("\nKesz.")


if __name__ == "__main__":
    main()
