"""
Letolti az osszes nezetet (fokep + altpic_1..4) mindharom termekhez.
Menti: static/glasses/raw/{id}_0.jpg, {id}_1.jpg, stb.
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
    {"id": "pepe",   "url": "https://eoptika.hu/pepe-jeans-szemuvegkeret-pj-3269-c1-52"},
    {"id": "reebok", "url": "https://eoptika.hu/reebok-szemuvegkeret-r-1017-03-52"},
    {"id": "guess",  "url": "https://eoptika.hu/guess-szemuvegkeret-gu-2707-n-056-51"},
]


def get_all_image_urls(page_url: str) -> list:
    req = urllib.request.Request(page_url, headers=HEADERS)
    html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

    # Osszes 1600x1200 kep URL (fokep + altpic-ok)
    all_matches = re.findall(
        r'(https://eoptika\.hu/img/\d+/[^"\']+/1600x1200,r/[^"\']+\.jpg)',
        html
    )
    # Deduplikalas, sorrend megtartasa
    seen = set()
    urls = []
    for m in all_matches:
        if m not in seen:
            seen.add(m)
            urls.append(m)
    return urls


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        data = urllib.request.urlopen(req).read()
        dest.write_bytes(data)
        print(f"    OK: {dest.name} ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"    HIBA: {e}")
        return False


def main():
    for p in PRODUCTS:
        print(f"\n[{p['id']}]")
        urls = get_all_image_urls(p["url"])
        print(f"  {len(urls)} kep talalt:")
        for i, url in enumerate(urls):
            label = "FOKEP" if "altpic" not in url else f"altpic_{url.split('altpic_')[1].split('/')[0]}"
            print(f"  [{i}] {label}: {url}")
            dest = RAW_DIR / f"{p['id']}_{i}.jpg"
            download(url, dest)

    print("\nKesz. Nezd meg a static/glasses/raw/ mappat.")


if __name__ == "__main__":
    main()
