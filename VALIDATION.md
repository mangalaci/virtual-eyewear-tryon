# Product Validation: Virtual Eyewear Try-On
### Finding First Customer — Positive Deviance Analysis
*Date: 2026-03-11*

## Product
Browser-based AR webcam try-on widget for eyewear e-commerce stores.

## Target Customer
Eyewear e-commerce store owners (Shopify etc.) — NOT end consumers.

---

## Workaround Inventory

| # | Observed Behavior | Workaround Method | Tier | Cost |
|---|---|---|---|---|
| 1 | "Home try-on" programs | Ship 5 frames, return 4 | **3 (paid)** | ~$15–25/shipment × return rate |
| 2 | Pay for Virtually.me / Ditto / GlassesUSA widget | Subscribe to generic SaaS | **3 (paid)** | $99–500/mo |
| 3 | Staff manually Photoshop glasses onto customer selfies | Manual image editing | **2 (embarrassment)** | 15 min/request |
| 4 | "Face shape guide" blog + quiz | Static content | **1** | Staff time, low conversion |
| 5 | Liberal "order 2, return 1" policy | Absorb 30–40% return rate | **3 (paid)** | Direct margin loss |

**Strongest signals: #2 and #5** — stores already paying, already losing money → build_now override.

---

## Test Results

| Test | Result |
|------|--------|
| Workaround exists? | YES |
| Paid workaround? | YES (Tier 3) → build_now override |
| Independent parallel invention? | YES — multiple stores independently built Photoshop workaround |
| Urgency this week? | YES — stores paying $200/mo for bad SaaS right now |
| Switching friction | Pure subtraction for SaaS users (swap embed script) |
| 10x delta vs manual Photoshop | YES (instant vs 15 min/request) |

---

## Workaround Cost Calculation

```
Home try-on:         $20/shipment × 40% return rate × 500 orders/mo = $4,000/mo
SaaS widget:         $99–500/mo + poor conversion
Photoshop workaround: 15 min × $25/hr × 50 requests/mo = $312/mo
─────────────────────────────────────────────────────────
Total monthly cost:  $500–4,500/mo depending on store size
Annualized:          $6,000–54,000/yr  ← price ceiling for annual contract
```

---

## First Customer Profile

```
Who:           Shopify eyewear store, 50–500 SKUs, currently paying for Virtually.me
               or complaining about high return rates in forums
Where to find: r/Entrepreneur, r/Shopify ("glasses" OR "eyewear" OR "returns"),
               Shopify Partner forums, Facebook "Shopify Entrepreneurs",
               ProductHunt reviews of Virtually.me / Ditto
Why now:       Current widget is generic, doesn't fit brand, customers find it clunky.
               Returns eating margin. Actively looking for alternatives.
DM script:     "I saw you're using [Virtually.me/Ditto] for try-on. I built a webcam-based
               AR try-on that works in the browser with no app install — I can integrate it
               on your store this week. Happy to do a free 2-week trial, then $X/mo if
               it reduces your return rate. Can I show you a 2-min demo?"
Offer:         Free 2-week integration → $149/mo after
```

---

## Verdict: INTERVIEW AND CHARGE

Paid workarounds confirmed. Parallel invention confirmed. Urgency real.

**Missing:** 3 confirmed deviants spoken to directly.

### Action This Week
1. Find 3 stores on Shopify currently using Virtually.me or complaining about returns
2. Send the DM script above
3. Offer free 2-week integration + $149/mo after
4. Embed the existing tool as a script tag (can be done manually)

**One charged card beats everything else.**
