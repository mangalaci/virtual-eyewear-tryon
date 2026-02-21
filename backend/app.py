import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
PRODUCTS_FILE = Path(__file__).resolve().parent / "products.json"

app = FastAPI(title="Virtual Eyewear Try-On")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/debug-paths")
async def debug_paths():
    static_dir = BASE_DIR / "static"
    glasses_dir = static_dir / "glasses"
    return {
        "BASE_DIR": str(BASE_DIR),
        "static_dir": str(static_dir),
        "static_exists": static_dir.exists(),
        "glasses_dir": str(glasses_dir),
        "glasses_exists": glasses_dir.exists(),
        "glasses_files": [f.name for f in glasses_dir.iterdir()] if glasses_dir.exists() else [],
    }


def load_products() -> list[dict]:
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class MeasurementInput(BaseModel):
    pd_mm: float
    face_width_mm: float
    bridge_width_mm: float


class ProductScore(BaseModel):
    id: str
    name: str
    score: float
    fit_notes: list[str]


@app.get("/glasses/{filename}")
def serve_glasses(filename: str):
    filepath = BASE_DIR / "static" / "glasses" / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    content = filepath.read_bytes()
    return Response(content=content, media_type="image/png")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/products")
async def get_products():
    return load_products()


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    products = load_products()
    for p in products:
        if p["id"] == product_id:
            return p
    raise HTTPException(status_code=404, detail="Product not found")


@app.post("/api/recommend", response_model=list[ProductScore])
async def recommend(measurements: MeasurementInput):
    products = load_products()
    recommended_lens_width = (measurements.face_width_mm - measurements.bridge_width_mm) / 2

    scored = []
    for p in products:
        fit_notes = []

        lens_diff = abs(recommended_lens_width - p["lens_width"]) / p["lens_width"]
        bridge_diff = abs(measurements.bridge_width_mm - p["bridge_width"]) / p["bridge_width"]
        temple_diff = abs(measurements.face_width_mm * 0.55 - p["temple_length"]) / p["temple_length"]

        score = 0.6 * (1 - min(lens_diff, 1)) + 0.3 * (1 - min(bridge_diff, 1)) + 0.1 * (1 - min(temple_diff, 1))

        if lens_diff < 0.08:
            fit_notes.append("Excellent lens width match")
        elif lens_diff < 0.15:
            fit_notes.append("Good lens width match")
        else:
            fit_notes.append("Lens width may not be ideal")

        if bridge_diff < 0.1:
            fit_notes.append("Great bridge fit")
        elif bridge_diff > 0.2:
            fit_notes.append("Bridge width differs significantly")

        scored.append(ProductScore(id=p["id"], name=p["name"], score=round(score, 3), fit_notes=fit_notes))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored
