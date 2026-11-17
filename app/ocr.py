"""EasyOCR wrapper. The reader is heavy (~100 MB models), so it is created once."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import OCR_LANGS
from .menu_parser import OcrBox


@lru_cache(maxsize=1)
def _reader():
    import easyocr  # imported lazily so tests and rules-only mode work without it
    return easyocr.Reader(OCR_LANGS, gpu=False)


def read_image(image: np.ndarray) -> list[OcrBox]:
    """`image` is an RGB numpy array."""
    results = _reader().readtext(image)  # [(points, text, conf), ...]
    return [OcrBox(points=[tuple(p) for p in pts], text=text, conf=float(conf))
            for pts, text, conf in results]
