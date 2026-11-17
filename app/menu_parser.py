"""Turn raw OCR boxes into menu items (name + price).

OCR returns words/lines with bounding boxes in random order. A menu row is
usually "dish name ........ 9,000". We group boxes whose vertical centres
are close into one row, read the row left-to-right and pull out the price.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

PRICE_RE = re.compile(r"(?:₩\s*)?(\d{1,3}(?:[,.]\d{3})+|\d{4,6})\s*(?:원|won|₩)?", re.IGNORECASE)
HAS_LETTER_RE = re.compile(r"[가-힣A-Za-z]")


@dataclass
class OcrBox:
    """One OCR detection: 4 corner points, text and confidence."""
    points: list[tuple[float, float]]
    text: str
    conf: float

    @property
    def x(self) -> float:
        return min(p[0] for p in self.points)

    @property
    def y_center(self) -> float:
        ys = [p[1] for p in self.points]
        return (min(ys) + max(ys)) / 2

    @property
    def height(self) -> float:
        ys = [p[1] for p in self.points]
        return max(ys) - min(ys)


@dataclass
class MenuItem:
    name: str
    price: int | None
    conf: float


def group_rows(boxes: list[OcrBox], min_conf: float = 0.2) -> list[list[OcrBox]]:
    boxes = [b for b in boxes if b.conf >= min_conf and b.text.strip()]
    if not boxes:
        return []
    tolerance = median(b.height for b in boxes) * 0.6
    rows: list[list[OcrBox]] = []
    for box in sorted(boxes, key=lambda b: b.y_center):
        if rows and abs(rows[-1][-1].y_center - box.y_center) <= tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [sorted(r, key=lambda b: b.x) for r in rows]


def parse_price(text: str) -> tuple[int | None, str]:
    """Return (price, text_without_price)."""
    matches = list(PRICE_RE.finditer(text))
    if not matches:
        return None, text
    m = matches[-1]                       # prices are usually at the end of a row
    price = int(re.sub(r"[,.]", "", m.group(1)))
    rest = (text[: m.start()] + text[m.end():]).strip(" .·-:")
    return price, rest


def extract_items(boxes: list[OcrBox]) -> list[MenuItem]:
    items: list[MenuItem] = []
    for row in group_rows(boxes):
        text = " ".join(b.text.strip() for b in row)
        price, name = parse_price(text)
        name = re.sub(r"[.…·]{2,}", " ", name)          # dot leaders between name and price
        name = re.sub(r"\s{2,}", " ", name).strip(" .…·-:")
        if len(name) < 2 or not HAS_LETTER_RE.search(name):
            continue                       # skip noise, page numbers, lone prices
        conf = sum(b.conf for b in row) / len(row)
        items.append(MenuItem(name=name, price=price, conf=round(conf, 3)))
    # On a priced menu, rows without a price are section headings ("샐러드", "MENU").
    if sum(1 for i in items if i.price) >= 2:
        items = [i for i in items if i.price]
    return items
