"""Step 2 — drop broken/duplicate images and split into train/val/test.

    python training/split_dataset.py --val 0.15 --test 0.15

data/raw/<class>/*  ->  data/split/{train,val,test}/<class>/*
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def valid_images(folder: Path):
    seen = set()
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in EXT:
            continue
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        if digest in seen:
            continue                                   # exact duplicate
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                if min(im.size) < 96:
                    continue                           # too small to be useful
        except Exception:
            continue                                   # broken file
        seen.add(digest)
        yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = ROOT / "split"
    shutil.rmtree(out, ignore_errors=True)
    rng = random.Random(args.seed)
    for cls_dir in sorted(p for p in (ROOT / "raw").iterdir() if p.is_dir()):
        files = list(valid_images(cls_dir))
        rng.shuffle(files)
        n_val, n_test = int(len(files) * args.val), int(len(files) * args.test)
        parts = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
        for part, items in parts.items():
            dst = out / part / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, dst / f.name)
        print(f"{cls_dir.name:15s} train={len(parts['train']):4d} val={len(parts['val']):3d} test={len(parts['test']):3d}")


if __name__ == "__main__":
    main()
