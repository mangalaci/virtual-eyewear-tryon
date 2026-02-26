"""
Process existing glasses PNGs: flood-fill lens area from centre → make transparent.
Prints sampled lens colours for products.json.
"""
from pathlib import Path
from collections import deque
import json
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
PRODUCTS_FILE = BASE / "products.json"

DARK_THR = 200  # sum(R+G+B) <= this = dark border pixel, flood-fill stops here


def process_png(path: Path) -> tuple[list, list] | None:
    img = Image.open(path).convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    h, w = rgba.shape[:2]
    opaque = rgba[:, :, 3] > 128

    # Hinge detection (same algorithm as JS analyzeGlassesImage)
    col_h = opaque.sum(axis=0)
    max_h = int(col_h.max())
    if max_h == 0:
        print("  No opaque pixels, skipping.")
        return None

    thr = max_h * 0.5
    left_hinge  = next((x for x in range(w // 2)       if col_h[x] > thr), 0)
    right_hinge = next((x for x in range(w - 1, w // 2, -1) if col_h[x] > thr), w - 1)
    fp_w = right_hinge - left_hinge

    left_lcx  = int(left_hinge + fp_w * 0.26)
    right_lcx = int(left_hinge + fp_w * 0.74)
    lcy = int(h * 0.55)
    print(f"  size={w}x{h}  hinges={left_hinge}..{right_hinge}  "
          f"centres=({left_lcx},{lcy}) ({right_lcx},{lcy})")

    # Flood-fill from lens centres through non-dark opaque pixels
    lens_mask = np.zeros((h, w), dtype=bool)
    visited   = np.zeros((h, w), dtype=bool)

    def flood_fill(sx: int, sy: int):
        if not (0 <= sx < w and 0 <= sy < h):
            return
        if not opaque[sy, sx]:
            return
        if int(rgba[sy, sx, 0]) + int(rgba[sy, sx, 1]) + int(rgba[sy, sx, 2]) <= DARK_THR:
            return
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        while q:
            cy, cx = q.popleft()
            lens_mask[cy, cx] = True
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if opaque[ny, nx]:
                        s = int(rgba[ny, nx, 0]) + int(rgba[ny, nx, 1]) + int(rgba[ny, nx, 2])
                        if s > DARK_THR:
                            q.append((ny, nx))

    flood_fill(left_lcx, lcy)
    flood_fill(right_lcx, lcy)
    print(f"  lens pixels found: {lens_mask.sum()}")

    if not lens_mask.any():
        print("  WARNING: no lens pixels found — skipping transparency step.")
        return None

    # Sample lens colours from central strip BEFORE making transparent
    sX1 = int(left_hinge + fp_w * 0.35)
    sX2 = int(left_hinge + fp_w * 0.65)
    ys = np.argwhere(lens_mask)[:, 0]
    y_min, y_max = int(ys.min()), int(ys.max())
    lens_h = max(1, y_max - y_min)
    top_band = y_min + int(lens_h * 0.3)
    bot_band = y_min + int(lens_h * 0.7)

    def mean_rgb(region_mask, region_rgba):
        pix = region_rgba[region_mask]
        if len(pix) == 0:
            return None
        return [int(round(float(c))) for c in pix[:, :3].mean(axis=0)]

    top_color = mean_rgb(
        lens_mask[y_min:top_band + 1, sX1:sX2 + 1],
        rgba[y_min:top_band + 1, sX1:sX2 + 1],
    ) or [200, 150, 60]

    bot_color = mean_rgb(
        lens_mask[bot_band:y_max + 1, sX1:sX2 + 1],
        rgba[bot_band:y_max + 1, sX1:sX2 + 1],
    ) or [120, 80, 30]

    print(f"  top colour: {top_color}")
    print(f"  bot colour: {bot_color}")

    # Make the lens area fully transparent
    rgba[lens_mask, 3] = 0
    Image.fromarray(rgba, "RGBA").save(path, "PNG")
    print(f"  Saved: {path.name}")
    return top_color, bot_color


def main():
    products = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
    changed = False

    for p in products:
        png_path = BASE / "static" / "glasses" / p["image"]
        if not png_path.exists():
            print(f"\n[{p['id']}] PNG not found: {png_path}")
            continue

        print(f"\n[{p['id']}]")
        result = process_png(png_path)

        if result is not None:
            top_color, bot_color = result
            p["lens_top"] = top_color
            p["lens_bot"] = bot_color
            changed = True
        elif p.get("type") == "optical":
            # Optical: clear lens, no tint needed
            p["lens_top"] = None
            p["lens_bot"] = None
            changed = True

    if changed:
        PRODUCTS_FILE.write_text(
            json.dumps(products, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nUpdated {PRODUCTS_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()
