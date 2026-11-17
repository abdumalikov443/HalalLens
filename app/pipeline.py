"""Glue: image -> OCR -> menu items -> rules + Claude -> final verdicts."""
from __future__ import annotations

import io
import json
from functools import lru_cache

import numpy as np
from PIL import Image

from . import classifier, llm, ocr
from .config import DISHES_PATH, LLM_ENABLED, MIN_DISH_CONFIDENCE
from .halal.rules import Verdict, check_text
from .menu_parser import extract_items

VERDICT_BY_LABEL = {v.label: v for v in Verdict}


def load_image(data: bytes, min_side: int = 1600, max_side: int = 2400) -> Image.Image:
    """Resize so the long side is between min_side and max_side.

    Small screenshots of menus have tiny text that OCR misses; upscaling them
    (up to 3x) makes the small menu rows readable. Huge phone photos are
    shrunk to keep OCR fast.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    long_side = max(img.size)
    if long_side < min_side:
        scale = min(min_side / long_side, 3.0)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    elif long_side > max_side:
        img.thumbnail((max_side, max_side))
    return img


def scan_menu(data: bytes, read=ocr.read_image, analyze=llm.analyze_items, debug: bool = False) -> dict:
    """`read` and `analyze` are injectable so tests can run without EasyOCR/Claude."""
    image = load_image(data)
    boxes = read(np.array(image))
    items = extract_items(boxes)
    llm_items = analyze([i.name for i in items]) if (LLM_ENABLED and items) else None

    out = []
    for idx, item in enumerate(items):
        rule = check_text(item.name)
        verdict, reasons = rule.verdict, list(rule.reasons_uz)
        entry = {"name": item.name, "price": item.price, "ocr_conf": item.conf,
                 "name_en": None, "name_uz": None, "ingredients": []}
        ai = (llm_items or {}).get(idx)
        if ai:
            entry.update(name_en=ai.name_en, name_uz=ai.name_uz, ingredients=ai.typical_ingredients)
            ai_verdict = VERDICT_BY_LABEL[ai.verdict]
            # Rules can only be made stricter by the model, never relaxed.
            if ai_verdict > verdict:
                verdict = ai_verdict
            if ai.reason_uz and ai.reason_uz not in reasons:
                reasons.append(ai.reason_uz)
        entry.update(verdict=verdict.label, reasons=reasons or ["Go'sht yoki spirt topilmadi"],
                     source="rules+ai" if ai else "rules")
        out.append(entry)

    result = {"items": out, "ai_used": llm_items is not None,
              "summary": {v.label: sum(1 for e in out if e["verdict"] == v.label) for v in Verdict}}
    if debug:                              # what OCR actually read, top-to-bottom
        result["ocr"] = [{"text": b.text, "conf": round(b.conf, 2), "y": round(b.y_center)}
                         for b in sorted(boxes, key=lambda b: b.y_center)]
        result["image_size"] = image.size
    return result


@lru_cache(maxsize=1)
def _dishes() -> dict:
    return json.loads(DISHES_PATH.read_text(encoding="utf-8"))


def scan_dish(data: bytes, predict=classifier.predict) -> dict:
    preds = predict(load_image(data))
    best, conf = preds[0]
    top = [{"dish": d, "confidence": c} for d, c in preds]
    if conf < MIN_DISH_CONFIDENCE:
        return {"recognized": False, "top": top,
                "message": "Taomni ishonchli taniy olmadim — boshqa burchakdan suratga oling."}
    info = _dishes().get(best, {})
    return {"recognized": True, "dish": best, "confidence": conf, "top": top, **info}
