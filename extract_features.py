"""Step 1: turn a dataset of real/fake videos (or images) into a feature CSV.

Examples
  python extract_features.py --mode video --real_dir data/real --fake_dir data/fake --limit 150 --out features_video.csv
  python extract_features.py --mode image --real_dir data/real_images --fake_dir data/fake_images --limit 2000 --out features_image.csv
Label: 0 = real, 1 = fake.  'group' = text before first '_' in file name (keeps clips of the same
source identity together when splitting train/test, which prevents leakage on FF++ / Celeb-DF).
"""
import argparse, os, random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.features import analyze_video, analyze_image
from src.face import FaceTracker

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def collect(dirs, exts):
    return sorted(p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["video", "image"], required=True)
    ap.add_argument("--real_dir", nargs="+", required=True)
    ap.add_argument("--fake_dir", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max files per class (0 = all)")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"features_{a.mode}.csv"
    exts = VIDEO_EXT if a.mode == "video" else IMAGE_EXT

    rows, done = [], set()
    if os.path.exists(out):  # resume
        old = pd.read_csv(out)
        rows, done = old.to_dict("records"), set(old["path"])
        print(f"Resuming: {len(done)} files already processed")

    tracker = FaceTracker(static=True) if a.mode == "image" else None
    rng = random.Random(42)
    for label, dirs in ((0, a.real_dir), (1, a.fake_dir)):
        files = collect(dirs, exts)
        rng.shuffle(files)
        if a.limit:
            files = files[: a.limit]
        print(f"{'real' if label == 0 else 'fake'}: {len(files)} files")
        for i, p in enumerate(tqdm(files)):
            if str(p) in done:
                continue
            try:
                if a.mode == "video":
                    res = analyze_video(p, max_frames=a.max_frames)
                    feats = res["features"] if res else None
                else:
                    feats = analyze_image(p, tracker)
            except Exception as e:
                print("skip", p.name, e)
                continue
            if feats is None:
                continue
            rows.append({"path": str(p), "label": label, "group": p.stem.split("_")[0], **feats})
            if len(rows) % 20 == 0:
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    print(f"Saved {len(df)} rows -> {out}\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
