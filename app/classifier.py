"""Load the trained MobileNetV2 dish classifier and predict top-k dishes."""
from __future__ import annotations

import json
from functools import lru_cache

from PIL import Image

from .config import CLASSES_PATH, MODEL_PATH


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise ModelNotTrained(f"Model not found at {MODEL_PATH}. Run training/train.py first.")
    import torch
    from torchvision import models, transforms

    classes: list[str] = json.loads(CLASSES_PATH.read_text())
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf, classes


def predict(image: Image.Image, k: int = 3) -> list[tuple[str, float]]:
    model, tf, classes = _load()          # raises ModelNotTrained before torch is needed
    import torch

    with torch.no_grad():
        probs = torch.softmax(model(tf(image.convert("RGB")).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, k=min(k, len(classes)))
    return [(classes[i], round(float(p), 4)) for p, i in zip(top.values, top.indices)]
