"""Copy the 12 HalalLens classes out of a downloaded Korean food dataset.

    python training/import_aihub.py "D:/datasets/kfood" --per-class 600

Works with any folder layout: it walks the whole tree and treats every folder
whose name is one of the Korean dish names below as that class. So it does not
matter whether the dataset is  kfood/구이/삼겹살/*.jpg  or  kfood/삼겹살/*.jpg.
Images go to data/raw/<class>/ - the same place scrape.py writes to, so you
can mix both sources and then run split_dataset.py.
"""
import argparse
import random
import shutil
from pathlib import Path

# folder name in the dataset  ->  class id used by HalalLens
CLASS_MAP = {
    "비빔밥": "bibimbap",
    "불고기": "bulgogi",
    "삼겹살": "samgyeopsal",
    "제육볶음": "jeyuk_bokkeum",
    "떡볶이": "tteokbokki",
    "김치찌개": "kimchi_jjigae",
    "잡채": "japchae",
    "김밥": "gimbap",
    "짜장면": "jjajangmyeon",
    "닭갈비": "dakgalbi",
    "물냉면": "naengmyeon",
    "비빔냉면": "naengmyeon",
    "냉면": "naengmyeon",
    "순대": "sundae",
}
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="root folder of the unzipped dataset")
    ap.add_argument("--per-class", type=int, default=600, help="max images per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    found: dict[str, list[Path]] = {}
    for folder in args.source.rglob("*"):
        if folder.is_dir() and folder.name.strip() in CLASS_MAP:
            cls = CLASS_MAP[folder.name.strip()]
            found.setdefault(cls, []).extend(f for f in folder.iterdir() if f.suffix.lower() in EXT)

    rng = random.Random(args.seed)
    for cls in sorted(set(CLASS_MAP.values())):
        files = found.get(cls, [])
        rng.shuffle(files)
        dst = OUT / cls
        dst.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files[: args.per_class]):
            shutil.copy2(f, dst / f"aihub_{cls}_{i:04d}{f.suffix.lower()}")
        status = "" if files else "   <-- NOT FOUND: check the folder name, or use scrape.py for this class"
        print(f"{cls:15s} {min(len(files), args.per_class):5d} copied (of {len(files)}){status}")


if __name__ == "__main__":
    main()
