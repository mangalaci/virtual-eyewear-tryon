"""
Glasses image preprocessing pipeline (OpenCV-based, no PyTorch/ONNX needed):

1. HSV threshold: white/near-white background -> transparent
2. Morphological cleanup
3. Contour-hierarchy lens-hole fill: ensure the lens area inside the frame
   ring is fully transparent (optical glasses, clear lens removed with bg)
4. Save as RGBA PNG to static/glasses/

Run after scrape_eoptika.py.
"""

import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "static" / "glasses" / "raw"
OUT_DIR       = BASE_DIR / "static" / "glasses"
PRODUCTS_FILE = BASE_DIR / "products.json"

# HSV thresholds for "white/near-white" background
BG_V_MIN = 210   # brightness threshold (0-255)
BG_S_MAX = 30    # saturation threshold (0-255)


def remove_background(img_rgb: np.ndarray) -> np.ndarray:
    """
    Replace white/near-white pixels with transparency.
    Returns RGBA uint8 array.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    bg = (v > BG_V_MIN) & (s < BG_S_MAX)   # True = background
    fg = (~bg).astype(np.uint8) * 255        # foreground mask

    # Small open: remove isolated noise pixels (1-2 px bleed from bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Small close: heal tiny gaps inside the frame ring WITHOUT closing the
    # large lens hole (kernel is too small to bridge a lens-sized gap)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    rgba = np.dstack([img_rgb, fg])
    return rgba.astype(np.uint8)


def detect_hinges(opaque: np.ndarray) -> tuple[int, int]:
    """Column-height heuristic to find left/right hinge x positions."""
    w = opaque.shape[1]
    col_h = opaque.sum(axis=0)
    max_h = col_h.max()
    if max_h == 0:
        return 0, w - 1
    thr = max_h * 0.5
    left  = next((x for x in range(w // 2)         if col_h[x] > thr), 0)
    right = next((x for x in range(w - 1, w // 2, -1) if col_h[x] > thr), w - 1)
    return left, right


def fill_lens_holes(rgba: np.ndarray, left_h: int, right_h: int) -> int:
    """
    Use RETR_CCOMP contour hierarchy to find enclosed transparent regions
    inside the frame ring.  Those are lens holes that may have been left
    partially opaque by the morphological close step.  Make them transparent.
    Returns number of pixels cleared.
    """
    h, w = rgba.shape[:2]
    opaque_u8 = (rgba[:, :, 3] > 128).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        opaque_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or len(contours) == 0:
        return 0

    hierarchy = hierarchy[0]
    min_area = (right_h - left_h) * h * 0.005   # 0.5% of frame area

    lens_mask = np.zeros((h, w), dtype=np.uint8)
    for i, h_info in enumerate(hierarchy):
        parent = h_info[3]
        if parent < 0:
            continue  # outer contour, not a hole
        area = cv2.contourArea(contours[i])
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(contours[i])
        cx = x + cw // 2
        if left_h <= cx <= right_h:
            cv2.drawContours(lens_mask, [contours[i]], -1, 255, cv2.FILLED)

    count = int((lens_mask > 0).sum())
    if count > 0:
        rgba[lens_mask > 0, 3] = 0
    return count


def process_image(raw_path: Path, out_path: Path) -> bool:
    try:
        orig = Image.open(raw_path).convert("RGB")
        img_rgb = np.array(orig, dtype=np.uint8)
        h, w = img_rgb.shape[:2]

        # 1. Background removal
        rgba = remove_background(img_rgb)
        opaque = rgba[:, :, 3] > 128

        if not opaque.any():
            print("  WARNING: no foreground pixels after background removal")
            return False

        # 2. Hinge detection
        left_h, right_h = detect_hinges(opaque)
        fp_w = right_h - left_h
        print(f"  hinges={left_h}..{right_h}  frame width={fp_w}px")

        # 3. Lens-hole transparency
        n = fill_lens_holes(rgba, left_h, right_h)
        print(f"  Lens holes cleared: {n} px")

        # 4. Save
        Image.fromarray(rgba, "RGBA").save(out_path, "PNG")
        kb = out_path.stat().st_size // 1024
        print(f"  Saved: {out_path.name} ({kb} KB)")
        return True

    except Exception as exc:
        print(f"  ERROR: {exc}")
        import traceback; traceback.print_exc()
        return False


def main():
    products = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))

    done = skipped = failed = 0
    for p in products:
        out_path = OUT_DIR / p["image"]
        sku = p["image"].replace(".png", "")

        raw_candidates = sorted(RAW_DIR.glob(f"{sku}.*"))
        if not raw_candidates:
            print(f"[{p['id']}] No raw image -> skip")
            skipped += 1
            continue

        if out_path.exists():
            print(f"[{p['id']}] Already processed -> skip")
            skipped += 1
            continue

        raw_path = raw_candidates[0]
        print(f"\n[{p['id']}]  {raw_path.name}")
        if process_image(raw_path, out_path):
            done += 1
        else:
            failed += 1

    print(f"\nDone: {done} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
