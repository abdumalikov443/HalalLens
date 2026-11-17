import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Read halallens/.env so ANTHROPIC_API_KEY and HALALLENS_* settings work
# without exporting them in every terminal. Real environment variables win.
load_dotenv(BASE_DIR / ".env", override=False)

# Claude. The SDK reads ANTHROPIC_API_KEY itself; without it the app runs in
# rules-only mode (no translation / ingredient explanation).
CLAUDE_MODEL = os.getenv("HALALLENS_MODEL", "claude-opus-5-5")
LLM_ENABLED = os.getenv("HALALLENS_LLM", "1") == "1"

# Dish classifier trained by training/train.py
MODEL_PATH = Path(os.getenv("HALALLENS_MODEL_PATH", BASE_DIR / "models" / "dish_classifier.pt"))
CLASSES_PATH = MODEL_PATH.with_suffix(".classes.json")
DISHES_PATH = BASE_DIR / "app" / "dishes.json"
MIN_DISH_CONFIDENCE = float(os.getenv("HALALLENS_MIN_CONF", "0.55"))

OCR_LANGS = ["ko", "en"]
MAX_UPLOAD_MB = 8
