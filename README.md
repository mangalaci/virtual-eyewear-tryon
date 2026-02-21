# Virtual Eyewear Try-On

AR-powered virtual glasses try-on using webcam and MediaPipe face tracking.

## Features
- Real-time face landmark detection with MediaPipe
- 3D perspective transformation for realistic glasses overlay
- Ray-Ban product catalog (Clubmaster, Wayfarer, Erika)
- FastAPI backend with product management

## Tech Stack
- **Backend:** FastAPI, Python
- **Frontend:** Vanilla JS, MediaPipe
- **Computer Vision:** MediaPipe Face Landmarker
- **3D Graphics:** Canvas 2D with perspective transforms

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
cd backend
uvicorn main:app --reload

# Open browser
http://localhost:8000
```

## Project Structure
- `backend/` - FastAPI server and product data
- `static/` - Frontend HTML/CSS/JS
- `templates/` - Jinja2 templates
