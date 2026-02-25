"""
Scrape product images from eoptika.hu, remove white background, save as RGBA PNG.
"""
import re
import sys
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import io

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "hu-HU,hu;q=0.9,en;q=0.8",
}

PRODUCTS = [
    {
        "id": "rayban-clubmaster",
        "url": "https://eoptika.hu/ray-ban-szemuvegkeret-rx-5154-2000-51",
        "out": "static/glasses/rayban_clubmaster.png",
    },
    {
        "id": "rayban-wayfarer",
        "url": "https://eoptika.hu/ray-ban-napszemuveg-rb-2140-901-50",
        "out": "static/glasses/rayban_wayfarer.png",
    },
    {
        "id": "rayban-erika",
        "url": "https://eoptika.hu/ray-ban-napszemuveg-rb-4171-86513-54",
        "out": "static/glasses/rayban_erika.png",
    },
]

RESIZE_RE = re.compile(r"/\d+x\d+,r/")


def find_product_image_url(page_url: str) -> str | None:
    print(f"  Fetching page: {page_url}")
    r = requests.get(page_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Try <article> first (Unas platform product thumbnails)
    for tag in soup.find_all(["img", "source"]):
        src = tag.get("data-src") or tag.get("src") or ""
        if not src:
            continue
        # Skip tiny/icon images
        if any(x in src for x in ["logo", "icon", "banner", "pixel", "placeholder"]):
            continue
        if RESIZE_RE.search(src):
            full = RESIZE_RE.sub("/", src)
            print(f"  Found (data-src resize): {full}")
            return full
        if src.startswith("http") and any(
            ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]
        ):
            print(f"  Found (src): {src}")
            return src

    # Fallback: og:image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        print(f"  Found (og:image): {og['content']}")
        return og["content"]

    return None


def remove_background(img_bytes: bytes, threshold: int = 235) -> Image.Image:
    """Remove near-white background from a product photo."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.uint8)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Pixels near-white → transparent
    near_white = (r > threshold) & (g > threshold) & (b > threshold)

    # Also flood-fill from corners to catch backgrounds not perfectly white
    from PIL import ImageFilter
    # Slightly lower threshold for flood fill seed
    seed_mask = (r > 220) & (g > 220) & (b > 220)

    # Simple BFS from image corners
    h, w = near_white.shape
    visited = np.zeros((h, w), bool)
    queue = []
    for y in [0, h - 1]:
        for x in range(w):
            if seed_mask[y, x] and not visited[y, x]:
                queue.append((y, x))
                visited[y, x] = True
    for x in [0, w - 1]:
        for y in range(h):
            if seed_mask[y, x] and not visited[y, x]:
                queue.append((y, x))
                visited[y, x] = True

    while queue:
        cy, cx = queue.pop()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and seed_mask[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    # Union: near-white OR flood-fill region → transparent
    bg_mask = near_white | visited
    data[:, :, 3] = np.where(bg_mask, 0, 255)

    # Also remove semi-transparent grey remnants (reflections, shadows):
    # any pixel whose RGB average is > 160 AND alpha is still set → make transparent
    grey_remnant = (r.astype(int) + g.astype(int) + b.astype(int)) / 3 > 160
    data[:, :, 3] = np.where(grey_remnant & (data[:, :, 3] > 0), 0, data[:, :, 3])

    result = Image.fromarray(data, "RGBA")

    # Trim to tight bounding box of fully-opaque pixels (ignores semi-transparent halo)
    alpha = np.array(result)[:, :, 3]
    opaque = np.argwhere(alpha > 128)
    if len(opaque):
        y0, x0 = opaque.min(axis=0)
        y1, x1 = opaque.max(axis=0)
        result = result.crop((x0, y0, x1 + 1, y1 + 1))

    return result


def process(product: dict) -> bool:
    print(f"\n[{product['id']}]")
    img_url = find_product_image_url(product["url"])
    if not img_url:
        print("  ERROR: image URL not found")
        return False

    print(f"  Downloading image...")
    try:
        r = requests.get(img_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return False

    print(f"  Downloaded {len(r.content)//1024} KB, removing background...")
    try:
        result = remove_background(r.content)
    except Exception as e:
        print(f"  ERROR in background removal: {e}")
        return False

    result.save(product["out"], "PNG")
    w, h = result.size
    import os
    size_kb = os.path.getsize(product["out"]) // 1024
    print(f"  Saved {product['out']}  ({w}x{h}, {size_kb} KB)")
    return True


if __name__ == "__main__":
    ok = 0
    for p in PRODUCTS:
        if process(p):
            ok += 1
    print(f"\nDone: {ok}/{len(PRODUCTS)} images saved.")
