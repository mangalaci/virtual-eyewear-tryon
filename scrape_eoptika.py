"""
eoptika.hu szemuvegkeret scraper
- Lazy-loading (data-src) kepeket ker le
- Letolti a static/glasses/ mappaba
- Frissiti a products.json-t

Hasznalat:
  python scrape_eoptika.py
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent
GLASSES_DIR = BASE_DIR / "static" / "glasses"
PRODUCTS_FILE = BASE_DIR / "backend" / "products.json"

GLASSES_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "hu-HU,hu;q=0.9,en-US;q=0.8",
    "Referer": "https://eoptika.hu/",
}

# Kategoria URL-ek -> kerettipus + meretadatok
CATEGORIES = [
    {
        "url": "https://eoptika.hu/kategoriak/szemuveg-akcio/cat-eye-keretek.html",
        "type": "cat-eye",
        "lens_width": 52,
        "bridge_width": 17,
        "temple_length": 140,
        "color": "#800020",
    },
    {
        "url": "https://eoptika.hu/kategoriak/szemuveg-akcio/fekete-szemuvegkeret.html",
        "type": "rectangular",
        "lens_width": 54,
        "bridge_width": 18,
        "temple_length": 140,
        "color": "#1a1a1a",
    },
    {
        "url": "https://eoptika.hu/kategoriak/szemuveg-akcio/sztk-retro-keret.html",
        "type": "retro",
        "lens_width": 50,
        "bridge_width": 20,
        "temple_length": 145,
        "color": "#8B4513",
    },
    {
        "url": "https://eoptika.hu/kategoriak/szemuveg-akcio/viztiszta-transzparens-keretek.html",
        "type": "transparent",
        "lens_width": 52,
        "bridge_width": 18,
        "temple_length": 140,
        "color": "#cccccc",
    },
    {
        "url": "https://eoptika.hu/kategoriak/szemuveg-akcio/nagymeretu-szemuvegkeret.html",
        "type": "oversized",
        "lens_width": 58,
        "bridge_width": 20,
        "temple_length": 150,
        "color": "#222222",
    },
]

MAX_PER_CATEGORY = 2

session = requests.Session()
session.headers.update(HEADERS)


def extract_size_from_sku(sku: str) -> dict:
    """
    Peldaul: GU_2903_028_52  ->  lens_width=52
    Az utolso szam a lencse szelessege mm-ben.
    """
    parts = sku.split("_")
    try:
        size = int(parts[-1])
        if 40 <= size <= 65:
            return {"lens_width": size}
    except (ValueError, IndexError):
        pass
    return {}


def scrape_category(cat: dict) -> list[dict]:
    """Egy kategoria oldalrol kiszedi a termekeket."""
    url = cat["url"]
    print(f"\n[{cat['type'].upper()}] {url}")

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        print(f"  HTTP {resp.status_code} - {len(resp.content) // 1024} KB")
    except Exception as e:
        print(f"  HIBA: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    products = []

    # Termek kartyak: article elemek js-product osztalyal
    articles = soup.select("article.product, article.js-product, [class*='page_artlist_sku']")
    print(f"  Talalt termek kartyak: {len(articles)}")

    for article in articles[:MAX_PER_CATEGORY]:
        # Nev: data-name attribútumból
        img_outer = article.select_one("[data-name]")
        name = img_outer["data-name"].strip() if img_outer else ""

        # Kepek: data-src attribútumban (lazy loading)
        img = article.select_one("img[data-src]")
        if not img:
            img = article.select_one("img[src]")
        if not img:
            continue

        img_url = img.get("data-src") or img.get("src") or ""
        if not img_url:
            continue
        if img_url.startswith("/"):
            img_url = urljoin("https://eoptika.hu", img_url)
        # Teljes meretu kep: "340x255,r/" reszt eltavolitjuk
        # pl. /img/43321/SKU/340x255,r/SKU.webp -> /img/43321/SKU/SKU.webp
        img_url = re.sub(r'/\d+x\d+,r/', '/', img_url)
        img_url = re.sub(r'\?.*$', '', img_url)  # query string eltavolitasa

        # SKU kinyerese az ID-bol (page_artlist_artlist_GU_2903_028_52)
        article_id = article.get("id", "")
        sku = article_id.replace("page_artlist_artlist_", "").strip()

        # Termek link
        a_tag = article.select_one("a[href]")
        href = urljoin(url, a_tag["href"]) if a_tag else ""

        # Meret az SKU-bol
        size_override = extract_size_from_sku(sku)

        if not name:
            name = img.get("alt") or sku or f"eOptika {cat['type']}"
        name = name.replace("(optikai keret)", "").replace("Noi szemuvegkeret", "").strip()

        products.append({
            "img_url": img_url,
            "name": name[:60],
            "sku": sku,
            "href": href,
            "size_override": size_override,
        })

    return products


def download_image(img_url: str, filepath: Path, referer: str = "") -> bool:
    try:
        hdrs = dict(HEADERS)
        if referer:
            hdrs["Referer"] = referer
        resp = session.get(img_url, headers=hdrs, timeout=20, stream=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "image" not in ct and "octet" not in ct:
            print(f"  NEM KEP: {ct}")
            return False
        filepath.write_bytes(resp.content)
        print(f"  LETOLTVE: {filepath.name} ({filepath.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  LETOLTESI HIBA: {e}")
        return False


def merge_products(new_products: list[dict]) -> int:
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {p["id"] for p in existing}
    added = 0
    for p in new_products:
        if p["id"] not in existing_ids:
            existing.append(p)
            added += 1
    with open(PRODUCTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return added


def main():
    print("=" * 50)
    print("eOptika.hu Scraper")
    print("=" * 50)
    print(f"Cel mappa: {GLASSES_DIR}")

    all_new_products = []

    for cat in CATEGORIES:
        items = scrape_category(cat)

        for i, item in enumerate(items):
            img_url = item["img_url"]
            name = item["name"]
            prod_id = f"eoptika-{cat['type']}-{i + 1}"

            # Fajlkiterjesztes
            ext_match = re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", img_url, re.IGNORECASE)
            ext = ext_match.group(1).lower() if ext_match else "webp"
            filename = f"eoptika_{cat['type']}_{i + 1}.{ext}"
            filepath = GLASSES_DIR / filename

            print(f"\n  [{i + 1}] {name}")
            print(f"  SKU: {item['sku']}")
            print(f"  Kep: {img_url[:90]}")

            if download_image(img_url, filepath, referer=cat["url"]):
                merged_sizes = {
                    "lens_width": cat["lens_width"],
                    "bridge_width": cat["bridge_width"],
                    "temple_length": cat["temple_length"],
                }
                merged_sizes.update(item["size_override"])

                all_new_products.append({
                    "id": prod_id,
                    "name": name,
                    "type": cat["type"],
                    "image": filename,
                    "color": cat["color"],
                    "description": f"eOptika.hu - {name}",
                    "source_url": item["href"],
                    **merged_sizes,
                })

        time.sleep(1)

    print("\n" + "=" * 50)
    if all_new_products:
        added = merge_products(all_new_products)
        print(f"OK: {len(all_new_products)} kep letoltve, {added} termek hozzaadva")
        print("\nLetoltott termekek:")
        for p in all_new_products:
            print(f"  - {p['name']} -> {p['image']}")
    else:
        print("Nem sikerult termekeket letolteni.")


if __name__ == "__main__":
    main()
