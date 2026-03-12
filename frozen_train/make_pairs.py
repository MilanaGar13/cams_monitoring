# make_pairs.py
import os
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

SRC = Path("collected_frames")
OUT = Path("dataset")
OUT.mkdir(exist_ok=True)
(OUT/"train"/"frozen").mkdir(parents=True, exist_ok=True)
(OUT/"train"/"normal").mkdir(parents=True, exist_ok=True)
(OUT/"val"/"frozen").mkdir(parents=True, exist_ok=True)
(OUT/"val"/"normal").mkdir(parents=True, exist_ok=True)

files = sorted([p for p in SRC.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])
n = len(files)
print("Found", n, "frames")

def augment_frozen(img: Image.Image):
    # small noise, small jpeg artifacts simulation, brightness jitter
    if random.random() < 0.4:
        # brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
    return img

def save_pair(a: Image.Image, b: Image.Image, out_path: Path):
    # concat two images side-by-side into one file? We prefer saving two files named similarly
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a.save(str(out_path.with_name(out_path.stem + "_a.png")))
    b.save(str(out_path.with_name(out_path.stem + "_b.png")))

idx = 0
# Create train/val split (90/10)
val_start = int(0.9 * n)

for i in tqdm(range(n-30)):
    # normal: pair (i, i+delta) with delta = 5,15,30
    deltas = [5,15,30]
    for d in deltas:
        j = i + d
        if j >= n: continue
        img_a = Image.open(files[i]).convert("RGB")
        img_b = Image.open(files[j]).convert("RGB")
        target_dir = "val" if i >= val_start else "train"
        save_pair(img_a, img_b, OUT/target_dir/"normal"/f"normal_{idx:06d}.png")
        idx += 1

    # frozen: take i and duplicate with small augmentations
    img = Image.open(files[i]).convert("RGB")
    img_f = augment_frozen(img.copy())
    target_dir = "val" if i >= val_start else "train"
    save_pair(img, img_f, OUT/target_dir/"frozen"/f"frozen_{idx:06d}.png")
    idx += 1

print("Done. Pairs written. Example counts:")
import glob
print("train frozen:", len(glob.glob("dataset/train/frozen/*_a.png")))
print("train normal:", len(glob.glob("dataset/train/normal/*_a.png")))

