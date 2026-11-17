"""Pipeline + API tests with OCR, Claude and the classifier replaced by fakes."""
import io

from fastapi.testclient import TestClient
from PIL import Image

from app import pipeline
from app.llm import LlmItem
from app.main import app
from tests.test_menu_parser import box


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, "PNG")
    return buf.getvalue()


def fake_read(_img):
    return [box(10, 10, "떡볶이"), box(200, 10, "5,000"), box(10, 50, "불고기 정식"), box(200, 50, "12,000")]


def test_scan_menu_rules_only():
    out = pipeline.scan_menu(png_bytes(), read=fake_read, analyze=lambda names: None)
    assert out["ai_used"] is False
    verdicts = {i["name"]: i["verdict"] for i in out["items"]}
    assert verdicts == {"떡볶이": "likely_ok", "불고기 정식": "doubtful"}


def test_ai_can_make_verdict_stricter_but_not_looser():
    def fake_ai(names):
        return {
            0: LlmItem(index=0, name_en="Tteokbokki", name_uz="Tokbokki", typical_ingredients=["rice cake"],
                       verdict="doubtful", reason_uz="Ba'zan go'shtli sho'rva."),
            1: LlmItem(index=1, name_en="Bulgogi set", name_uz="Bulgogi", typical_ingredients=["beef"],
                       verdict="likely_ok", reason_uz="Model xato qildi"),
        }
    items = pipeline.scan_menu(png_bytes(), read=fake_read, analyze=fake_ai)["items"]
    assert items[0]["verdict"] == "doubtful"        # raised by AI
    assert items[1]["verdict"] == "doubtful"        # rules keep it, AI cannot relax it
    assert items[0]["name_uz"] == "Tokbokki"


def test_scan_dish_low_confidence():
    out = pipeline.scan_dish(png_bytes(), predict=lambda img: [("bulgogi", 0.31), ("japchae", 0.2)])
    assert out["recognized"] is False


def test_scan_dish_known():
    out = pipeline.scan_dish(png_bytes(), predict=lambda img: [("samgyeopsal", 0.93)])
    assert out["recognized"] and out["verdict"] == "haram"


def test_api_rejects_non_images():
    client = TestClient(app)
    r = client.post("/api/scan-menu", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_api_dish_without_model_returns_503():
    client = TestClient(app)
    r = client.post("/api/scan-dish", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 503


def test_small_images_are_upscaled_for_ocr():
    buf = io.BytesIO()
    Image.new("RGB", (400, 600), "white").save(buf, "PNG")
    img = pipeline.load_image(buf.getvalue())
    assert max(img.size) == 1600


def test_debug_returns_raw_ocr():
    out = pipeline.scan_menu(png_bytes(), read=fake_read, analyze=lambda n: None, debug=True)
    assert [o["text"] for o in out["ocr"]][:2] == ["떡볶이", "5,000"]
