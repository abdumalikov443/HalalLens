"""Step 1 — download raw images for every dish class (runs on your own PC).

    python training/scrape.py --per-query 120 --max-per-class 300
    python training/scrape.py --only sundae gimbap          # just some classes

Each class is searched with several queries (Korean, English, descriptive),
exact duplicates are removed, and the result lands in data/raw/<class>/.
Re-running is safe: images already in the folder are kept and not duplicated.

Scraped images are noisy: ALWAYS open the folders afterwards and delete
wrong photos by hand (logos, menus, people, other dishes). That cleaning
step decides your final accuracy.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from bing_image_downloader import downloader

QUERIES = {
    "bibimbap":      ["비빔밥", "bibimbap", "돌솥비빔밥", "korean bibimbap bowl"],
    "bulgogi":       ["불고기", "bulgogi", "소불고기", "korean beef bulgogi dish"],
    "samgyeopsal":   ["삼겹살 구이", "samgyeopsal", "삼겹살 불판", "korean pork belly bbq"],
    "jeyuk_bokkeum": ["제육볶음", "jeyuk bokkeum", "제육볶음 정식", "spicy stir fried pork korean"],
    "tteokbokki":    ["떡볶이", "tteokbokki", "국물떡볶이", "korean spicy rice cakes"],
    "kimchi_jjigae": ["김치찌개", "kimchi jjigae", "김치찌개 뚝배기", "kimchi stew korean"],
    "japchae":       ["잡채", "japchae", "잡채 요리", "korean glass noodles japchae"],
    "gimbap":        ["김밥", "gimbap", "김밥 한줄", "korean kimbap rolls"],
    "jjajangmyeon":  ["짜장면", "jjajangmyeon", "짜장면 그릇", "korean black bean noodles"],
    "dakgalbi":      ["닭갈비", "dakgalbi", "춘천닭갈비", "korean spicy chicken dakgalbi"],
    "naengmyeon":    ["냉면", "naengmyeon", "물냉면", "비빔냉면"],
    "sundae":        ["순대", "sundae korean food", "순대 접시", "korean blood sausage sundae"],
}

ROOT = Path(__file__).resolve().parent.parent / "data"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect(cls: str, queries: list[str], per_query: int, max_per_class: int) -> int:
    dst = ROOT / "raw" / cls
    dst.mkdir(parents=True, exist_ok=True)
    seen = {md5(f) for f in dst.iterdir() if f.suffix.lower() in EXT}
    count = len(seen)
    tmp = ROOT / "_download"
    for query in queries:
        if count >= max_per_class:
            break
        print(f"   query: {query}")
        try:
            downloader.download(query, limit=per_query, output_dir=str(tmp), adult_filter_off=False,
                                force_replace=False, timeout=15, verbose=False)
        except Exception as e:                       # network hiccups should not stop the run
            print(f"   ! skipped ({e})")
            continue
        for f in sorted((tmp / query).glob("*")):
            if f.suffix.lower() not in EXT or count >= max_per_class:
                continue
            h = md5(f)
            if h in seen:
                continue                             # same image from another query
            seen.add(h)
            shutil.move(str(f), dst / f"{cls}_{h[:10]}{f.suffix.lower()}")
            count += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=120)
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--only", nargs="*", help="class names to download (default: all)")
    args = ap.parse_args()
    for cls, queries in QUERIES.items():
        if args.only and cls not in args.only:
            continue
        print(f"== {cls}")
        n = collect(cls, queries, args.per_query, args.max_per_class)
        print(f"   {n} images in data/raw/{cls}")


if __name__ == "__main__":
    main()
