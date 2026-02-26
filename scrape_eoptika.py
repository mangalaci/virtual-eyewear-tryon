"""
eoptika.hu szemüvegkeret scraper
- Kategória oldalakról szedi a termékeket (MAX_PER_CATEGORY db)
- Minden terméklapról kiszedi az egyedi méreteket
- Letölti a képeket static/glasses/raw/ mappába
- Frissíti a products.json-t
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "static" / "glasses" / "raw"
PRODUCTS_FILE = BASE_DIR / "products.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "hu-HU,hu;q=0.9,en-US;q=0.8",
    "Referer": "https://eoptika.hu/",
}

CATEGORIES = [
    {"url": "https://eoptika.hu/kategoriak/szemuveg-akcio/cat-eye-keretek.html",           "type": "cat-eye"},
    {"url": "https://eoptika.hu/kategoriak/szemuveg-akcio/fekete-szemuvegkeret.html",       "type": "rectangular"},
    {"url": "https://eoptika.hu/kategoriak/szemuveg-akcio/sztk-retro-keret.html",           "type": "retro"},
    {"url": "https://eoptika.hu/kategoriak/szemuveg-akcio/viztiszta-transzparens-keretek.html", "type": "transparent"},
    {"url": "https://eoptika.hu/kategoriak/napszemuveg/ferfi-napszemuveg.html",             "type": "sunglasses"},
]

MAX_PER_CATEGORY = 10

session = requests.Session()
session.headers.update(HEADERS)


def extract_image_url(article) -> str:
    """
    <picture> elemből szedi ki a képet:
      <source data-srcset="URL 1x, URL2x" ...>
    A teljes méretű képet adja vissza (eltávolítja a /NxM,r/ méretprefixet).
    """
    source = article.select_one("picture source[data-srcset]")
    if source:
        srcset = source.get("data-srcset", "")
        # első URL (1x verzió), eltávolítjuk a ' 1x' részt
        raw_url = srcset.split(",")[0].strip().split(" ")[0]
    else:
        img = article.select_one("img[data-src]") or article.select_one("img[src]")
        raw_url = (img.get("data-src") or img.get("src") or "") if img else ""

    if not raw_url or "space.gif" in raw_url:
        return ""

    url = re.sub(r"/\d+x\d+,r/", "/", raw_url)   # méretprefix eltávolítása
    url = re.sub(r"\?.*$", "", url)               # query string eltávolítása
    return url


def fetch_sizes(product_url: str) -> dict:
    """
    Terméklapról szedi ki a méreteket az li elemekből:
      'Lencse szélessége: 52 mm'
      'Hídsszélesség: 16 mm'
      'Szárhossz: 140 mm'
    """
    try:
        resp = session.get(product_url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"    Méretlekérés hiba: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    sizes = {}

    for li in soup.select("li"):
        text = li.get_text(strip=True)
        m = re.search(r"(\d+)\s*mm", text)
        if not m:
            continue
        val = int(m.group(1))
        tl = text.encode("ascii", "ignore").decode().lower()
        if "lencse sz" in tl and not sizes.get("lens_width"):
            sizes["lens_width"] = val
        elif ("hidsz" in tl or "h\xeddsz" in text.lower()) and not sizes.get("bridge_width"):
            sizes["bridge_width"] = val
        elif "rhossz" in tl and not sizes.get("temple_length"):
            sizes["temple_length"] = val

    return sizes


def scrape_category(cat: dict) -> list[dict]:
    url = cat["url"]
    print(f"\n[{cat['type'].upper()}] {url}")

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        print(f"  HTTP {resp.status_code} — {len(resp.content) // 1024} KB")
    except Exception as e:
        print(f"  HIBA: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    articles = soup.select("article.js-product, article.product")
    print(f"  Talált termékek: {len(articles)}")

    items = []
    for article in articles[:MAX_PER_CATEGORY]:
        img_url = extract_image_url(article)
        if not img_url:
            continue

        name_tag = article.select_one("[data-name]")
        name = name_tag["data-name"].strip() if name_tag else ""
        name = re.sub(r"\(optikai keret\)|\(napszemüveg\)|Női|Férfi|Unisex", "", name, flags=re.IGNORECASE).strip()

        sku = article.get("id", "").replace("page_artlist_artlist_", "").strip()

        a_tag = article.select_one("a.product_link_normal[href]")
        product_url = a_tag["href"] if a_tag else ""
        if product_url and not product_url.startswith("http"):
            product_url = "https://eoptika.hu" + product_url

        if not name:
            name = article.select_one("img[alt]")
            name = name["alt"] if name else sku

        items.append({
            "sku": sku,
            "name": name[:60],
            "img_url": img_url,
            "product_url": product_url,
        })

    return items


def download_image(img_url: str, filepath: Path) -> bool:
    try:
        resp = session.get(img_url, timeout=20, stream=True)
        resp.raise_for_status()
        if "image" not in resp.headers.get("content-type", ""):
            print(f"    Nem kép: {resp.headers.get('content-type')}")
            return False
        filepath.write_bytes(resp.content)
        print(f"    Letöltve: {filepath.name} ({filepath.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"    Letöltési hiba: {e}")
        return False


def load_existing_products() -> list[dict]:
    if PRODUCTS_FILE.exists():
        return json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
    return []


def main():
    print("=" * 60)
    print("eOptika.hu Scraper")
    print(f"Max {MAX_PER_CATEGORY} termék/kategória, {len(CATEGORIES)} kategória")
    print("=" * 60)

    existing = load_existing_products()
    existing_ids = {p["id"] for p in existing}
    new_products = []

    for cat in CATEGORIES:
        items = scrape_category(cat)

        for item in items:
            sku = item["sku"]
            prod_id = sku.lower().replace("_", "-")

            if prod_id in existing_ids:
                print(f"  [MÁR MEGVAN] {prod_id}")
                continue

            # Fájlkiterjesztés az URL-ből
            ext_m = re.search(r"\.(webp|jpg|jpeg|png)$", item["img_url"], re.IGNORECASE)
            ext = ext_m.group(1).lower() if ext_m else "webp"
            raw_filename = f"{sku}.{ext}"
            raw_path = RAW_DIR / raw_filename

            print(f"\n  [{sku}] {item['name']}")

            # Kép letöltése (ha még nincs meg)
            if raw_path.exists():
                print(f"    Már letöltve: {raw_filename}")
            else:
                if not download_image(item["img_url"], raw_path):
                    continue

            # Méretek lekérése a terméklapról
            sizes = {}
            if item["product_url"]:
                print(f"    Méretek: {item['product_url']}")
                sizes = fetch_sizes(item["product_url"])
                print(f"    -> {sizes}")
                time.sleep(0.5)

            # A feldolgozott PNG neve (SAM majd ide generálja)
            png_filename = f"{sku}.png"

            new_products.append({
                "id": prod_id,
                "name": item["name"],
                "type": cat["type"],
                "lens_width":    sizes.get("lens_width",    52),
                "bridge_width":  sizes.get("bridge_width",  18),
                "temple_length": sizes.get("temple_length", 140),
                "color":         "#1a1a1a",
                "image":         png_filename,
                "description":   item["name"],
                "source_url":    item["product_url"],
            })
            existing_ids.add(prod_id)

        time.sleep(1)

    if new_products:
        all_products = existing + new_products
        PRODUCTS_FILE.write_text(
            json.dumps(all_products, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n{'=' * 60}")
        print(f"Kész: {len(new_products)} uj termek hozzaadva -> products.json")
        print(f"Képek: {RAW_DIR}")
    else:
        print("\nNincs új termék.")


if __name__ == "__main__":
    main()
