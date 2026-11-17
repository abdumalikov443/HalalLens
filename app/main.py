"""HalalLens API.  Run:  uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from . import pipeline
from .classifier import ModelNotTrained
from .config import BASE_DIR, MAX_UPLOAD_MB  # loads .env first

logging.basicConfig(level=logging.INFO)
logging.getLogger("halallens").info(
    "Claude API key: %s", "found" if os.getenv("ANTHROPIC_API_KEY") else "NOT SET - rules-only mode")
app = FastAPI(title="HalalLens", version="0.1.0")
STATIC = BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


async def _read(file: UploadFile) -> bytes:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "Faqat rasm yuklang (jpg, png, webp).")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Rasm {MAX_UPLOAD_MB} MB dan katta.")
    return data


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/scan-menu")
async def scan_menu(file: UploadFile, debug: bool = False):
    try:
        return pipeline.scan_menu(await _read(file), debug=debug)
    except UnidentifiedImageError:
        raise HTTPException(400, "Rasmni o'qib bo'lmadi.")


@app.post("/api/scan-dish")
async def scan_dish(file: UploadFile):
    try:
        return pipeline.scan_dish(await _read(file))
    except ModelNotTrained as e:
        raise HTTPException(503, str(e))
    except UnidentifiedImageError:
        raise HTTPException(400, "Rasmni o'qib bo'lmadi.")
