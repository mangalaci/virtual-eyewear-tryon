"""
Letolti a 3 eoptika.hu szemuveg fotojet (szembol nezet, 1600x1200).
Menti: static/glasses/raw/ mappaba
"""

import re
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "static" / "glasses" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0",
    "Referer": "https://eoptika.hu/",
}

PRODUCTS = [
    {
        "id": "pepe",
        "url": "https://eoptika.hu/pepe-jeans-szemuvegkeret-pj-3269-c1-52",
        "name": "Pepe Jeans PJ 3269 C1 52",
    },
    {
        "id": "reebok",
        "url": "https://eoptika.hu/reebok-szemuvegkeret-r-1017-03-52",
        "name": "Reebok R 1017 03 52",
    },
    {
        "id": "guess",
        "url": "https://eoptika.hu/guess-szemuvegkeret-gu-2707-n-056-51",
        "name": "Guess GU 2707-N 056 51",
    },
]


def get_main_image_url(page_url: str) -> str:
    req = urllib.request.Request(page_url, headers=HEADERS)
    html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

    # Keresi a 1600x1200 fokepi URL-t (altpic NELKUL)
    matches = re.findall(
        r'(https://eoptika\.hu/img/\d+/[^/]+/1600x1200,r/[^"\']+\.jpg)',
        html
    )

    # Az altpic nelkuli az elso (fokep)
    main = [m for m in matches if "altpic" not in m]
    if main:
        return main[0]

    # Fallback: teljes meretu jpg altpic nelkul
    matches2 = re.findall(
        r'(https://eoptika\.hu/img/\d+/[^/]+/[^"\']+\.jpg)',
        html
    )
    main2 = [m for m in matches2 if "altpic" not in m and "," not in m]
    if main2:
        return main2[0]

    return ""


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        data = urllib.request.urlopen(req).read()
        dest.write_bytes(data)
        print(f"  OK: {dest.name} ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  HIBA: {e}")
        return False


def main():
    for p in PRODUCTS:
        print(f"\n[{p['id']}] {p['name']}")
        img_url = get_main_image_url(p["url"])
        if not img_url:
            print("  Nem talalt fokepi URL-t!")
            continue
        print(f"  URL: {img_url}")
        dest = RAW_DIR / f"{p['id']}.jpg"
        download(img_url, dest)

    print("\nKesz. Kepek: static/glasses/raw/")


if __name__ == "__main__":
    main()
