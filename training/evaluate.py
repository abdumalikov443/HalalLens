"""Step 4 — measure the saved model on the untouched test set.

    python training/evaluate.py

Prints overall + per-class accuracy and saves models/confusion_matrix.png.
These are the numbers you put in the portfolio.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models

from train import eval_tf

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "dish_classifier.pt"


def main():
    classes = json.loads(MODEL.with_suffix(".classes.json").read_text())
    ds = datasets.ImageFolder(ROOT / "data" / "split" / "test", eval_tf)
    assert ds.classes == classes, "test folders differ from training classes"
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    cm = np.zeros((len(classes), len(classes)), dtype=int)
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=32):
            for t, p in zip(y, model(x).argmax(1)):
                cm[t, p] += 1

    acc = np.trace(cm) / cm.sum()
    print(f"TEST accuracy: {acc:.2%} on {cm.sum()} images\n")
    for i, c in enumerate(classes):
        print(f"{c:15s} {cm[i, i] / max(cm[i].sum(), 1):6.1%}  ({cm[i].sum()} images)")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"HalalLens dish classifier — test acc {acc:.1%}")
    fig.tight_layout()
    fig.savefig(ROOT / "models" / "confusion_matrix.png", dpi=150)
    print("\nsaved models/confusion_matrix.png")


if __name__ == "__main__":
    main()
