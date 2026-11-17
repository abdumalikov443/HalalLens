"""Step 3 — fine-tune MobileNetV2 on the Korean dish dataset.

    python training/train.py --epochs 15

Phase 1 trains only the new classifier head (fast, stable).
Phase 2 unfreezes the last blocks with a small learning rate (better accuracy).
The best checkpoint by validation accuracy is saved to models/.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "split"
OUT = ROOT / "models" / "dish_classifier.pt"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = datasets.ImageFolder(DATA / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA / "val", eval_tf)
    assert train_ds.classes == val_ds.classes, "train/val class folders differ"
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, args.batch, num_workers=2)
    print(f"{len(train_ds)} train / {len(val_ds)} val images, {len(train_ds.classes)} classes, device={device}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    best_acc, history = 0.0, []
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_epochs + 1:                       # phase 2: fine-tune last blocks
            for p in model.features[-4:].parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            print("-- unfreezing last 4 feature blocks")
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, device, criterion)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        flag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), OUT)
            OUT.with_suffix(".classes.json").write_text(json.dumps(train_ds.classes))
            flag = "  <- saved"
        print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.2%}  val {va_loss:.3f}/{va_acc:.2%}  {time.time()-t0:.0f}s{flag}")

    (ROOT / "models" / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val accuracy: {best_acc:.2%}  ->  {OUT}")


if __name__ == "__main__":
    main()
