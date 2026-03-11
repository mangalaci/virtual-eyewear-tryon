"""
Integralja az uj eoptika szemuvegeket a virtual try-on-ba:
1. Lencse-kozeppontok auto-detektalasa az atlatszo teruletekbol
2. PNG masolasa processed/ mappaba
3. calibration.json frissitese
4. products.json frissitese
"""

import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

BASE_DIR      = Path(__file__).parent
GLASSES_DIR   = BASE_DIR / "static" / "glasses"
PROCESSED_DIR = GLASSES_DIR / "processed"
PRODUCTS_FILE = BASE_DIR / "backend" / "products.json"
CALIB_FILE    = PROCESSED_DIR / "calibration.json"

NEW_GLASSES = [
    {
        "id":             "pepe-pj3269-c1-52",
        "name":           "Pepe Jeans PJ 3269 C1",
        "type":           "optical",
        "lens_width":     52,
        "bridge_width":   18,
        "temple_length":  140,
        "frame_image":    "pepe_frame.png",
        "cal_key":        "pepe_pj3269",
        "description":    "Pepe Jeans PJ 3269 C1 52 optikai keret",
        "source_url":     "https://eoptika.hu/pepe-jeans-szemuvegkeret-pj-3269-c1-52",
        "src_png":        "pepe.png",
    },
    {
        "id":             "reebok-r1017-03-52",
        "name":           "Reebok R 1017 03",
        "type":           "optical",
        "lens_width":     52,
        "bridge_width":   17,
        "temple_length":  140,
        "frame_image":    "reebok_frame.png",
        "cal_key":        "reebok_r1017",
        "description":    "Reebok R 1017 03 52 optikai keret",
        "source_url":     "https://eoptika.hu/reebok-szemuvegkeret-r-1017-03-52",
        "src_png":        "reebok.png",
    },
    {
        "id":             "guess-gu2707n-056-51",
        "name":           "Guess GU 2707-N 056",
        "type":           "optical",
        "lens_width":     51,
        "bridge_width":   17,
        "temple_length":  135,
        "frame_image":    "guess_frame.png",
        "cal_key":        "guess_gu2707",
        "description":    "Guess GU 2707-N 056 51 optikai keret",
        "source_url":     "https://eoptika.hu/guess-szemuvegkeret-gu-2707-n-056-51",
        "src_png":        "guess.png",
    },
]


def find_lens_centers(png_path: Path):
    """
    Megtalalja a ket lencse-kozeppontot az atlatszo teruletekbol.
    Visszaad: (left_cx, left_cy, right_cx, right_cy, img_w, img_h)
    """
    img  = Image.open(png_path).convert("RGBA")
    data = np.array(img)
    img_h, img_w = data.shape[:2]

    # Atlatszo maszk: alpha < 30 = atlatszo
    alpha_mask = data[..., 3] < 30

    # Opak maszk: alpha >= 128 (keret)
    opaque_mask = data[..., 3] >= 128

    # Csak az opak terulet befoglalo teglalapjan beluli atlatszo pixelek
    rows = np.any(opaque_mask, axis=1)
    cols = np.any(opaque_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Atlatszo terulet a keret hatarlapan belul
    inner_transparent = alpha_mask.copy()
    inner_transparent[:rmin, :] = False
    inner_transparent[rmax+1:, :] = False
    inner_transparent[:, :cmin] = False
    inner_transparent[:, cmax+1:] = False

    # Cimkezett osszetevo reszek (lencse "lyukak")
    labeled, num_features = ndimage.label(inner_transparent)
    if num_features < 2:
        # Fallback: becsult poziciok a keret alapjan
        mid_y = (rmin + rmax) / 2
        q1_x  = cmin + (cmax - cmin) * 0.25
        q3_x  = cmin + (cmax - cmin) * 0.75
        print(f"  FIGYELEM: csak {num_features} atlatszo teruletet talalt, becslest hasznalok")
        return q1_x, mid_y, q3_x, mid_y, img_w, img_h

    # Lencse-terulet meret szerinti rendezes (legalacsonyabb 2 kihagyasa)
    sizes = ndimage.sum(inner_transparent, labeled, range(1, num_features + 1))
    sorted_idx = np.argsort(sizes)[::-1]

    # Ket legnagyobb teruletet vesszuk lencsenek
    lens_indices = sorted(sorted_idx[:2] + 1)  # 1-indexed labels

    centers = []
    for idx in lens_indices:
        ys, xs = np.where(labeled == idx)
        centers.append((float(np.mean(xs)), float(np.mean(ys))))

    # Balrol jobbra rendezve
    centers.sort(key=lambda c: c[0])
    (left_cx, left_cy), (right_cx, right_cy) = centers

    return left_cx, left_cy, right_cx, right_cy, img_w, img_h


def main():
    calib = json.loads(CALIB_FILE.read_text(encoding="utf-8"))

    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        products = json.load(f)
    existing_ids = {p["id"] for p in products}

    for g in NEW_GLASSES:
        src = GLASSES_DIR / g["src_png"]
        dst = PROCESSED_DIR / g["frame_image"]

        print(f"\n[{g['id']}]")

        if not src.exists():
            print(f"  HIANYZIK: {src}")
            continue

        # 1. Lencse-kozeppontok detektalasa
        print("  Lencse-kozeppontok detektalasa...")
        try:
            lx, ly, rx, ry, w, h = find_lens_centers(src)
            pd_px = float(np.sqrt((rx - lx)**2 + (ry - ly)**2))
            cx_frac = round((lx + rx) / 2 / w, 3)
            cy_frac = round((ly + ry) / 2 / h, 3)
            print(f"  Bal: ({lx:.0f}, {ly:.0f})  Jobb: ({rx:.0f}, {ry:.0f})")
            print(f"  pd_px={pd_px:.0f}  center=({cx_frac}, {cy_frac})  meret={w}x{h}")
        except Exception as e:
            print(f"  HIBA (detekcio): {e}")
            continue

        # 2. Kalibraciohoz mentese
        calib[g["cal_key"]] = {
            "left_cx": round(lx, 1), "left_cy": round(ly, 1),
            "right_cx": round(rx, 1), "right_cy": round(ry, 1),
            "img_w": w, "img_h": h,
            "pd_px": round(pd_px, 1),
            "lens_center_x_frac": cx_frac,
            "lens_center_y_frac": cy_frac,
        }

        # 3. PNG masolasa processed/-ba
        shutil.copy2(src, dst)
        print(f"  Masolva: {dst.name}")

        # 4. Termek hozzaadasa (ha meg nincs)
        if g["id"] not in existing_ids:
            entry = {k: v for k, v in g.items() if k != "src_png"}
            products.append(entry)
            print(f"  Termek hozzaadva: {g['id']}")
        else:
            print(f"  Termek mar letezik: {g['id']}")

    # Mentesek
    CALIB_FILE.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\ncalibration.json frissitve")

    with open(PRODUCTS_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print("products.json frissitve")

    print("\nKesz! Inditsd el a szervert es probald ki.")


if __name__ == "__main__":
    main()
