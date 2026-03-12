# make_view_dataset.py
import os
import cv2
import numpy as np
import random
from glob import glob

ANCHOR_DIR = "dataset_view/anchor"
POS_DIR = "dataset_view/positive"
NEG_DIR = "dataset_view/negative"
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

def random_affine(img, max_dx=0.05, max_dy=0.05, max_angle=2, max_scale=0.03):
    h, w = img.shape[:2]
    # translations (fraction)
    dx = int(random.uniform(-max_dx, max_dx) * w)
    dy = int(random.uniform(-max_dy, max_dy) * h)
    angle = random.uniform(-max_angle, max_angle)
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0,2] += dx
    M[1,2] += dy
    out = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    return out

def random_homography(img, max_perturb=0.12):
    # perturb corners
    h, w = img.shape[:2]
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = src + np.float32([
        [random.uniform(-max_perturb, max_perturb)*w, random.uniform(-max_perturb, max_perturb)*h],
        [random.uniform(-max_perturb, max_perturb)*w, random.uniform(-max_perturb, max_perturb)*h],
        [random.uniform(-max_perturb, max_perturb)*w, random.uniform(-max_perturb, max_perturb)*h],
        [random.uniform(-max_perturb, max_perturb)*w, random.uniform(-max_perturb, max_perturb)*h],
    ])
    H, _ = cv2.findHomography(src, dst)
    out = cv2.warpPerspective(img, H, (w,h), borderMode=cv2.BORDER_REFLECT)
    return out

def make_dataset_per_anchor(path, name_prefix, pos_count=200, neg_count=200):
    img = cv2.imread(path)
    if img is None:
        print("Cannot read", path); return
    for i in range(pos_count):
        out = random_affine(img, max_dx=0.03, max_dy=0.03, max_angle=1.0, max_scale=0.02)
        cv2.imwrite(f"{POS_DIR}/{name_prefix}_pos_{i:04d}.jpg", out)
    for i in range(neg_count):
        if random.random() < 0.6:
            out = random_homography(img, max_perturb=0.2)
        else:
            out = random_affine(img, max_dx=0.2, max_dy=0.2, max_angle=random.uniform(5,25), max_scale=random.uniform(0.05,0.4))
        cv2.imwrite(f"{NEG_DIR}/{name_prefix}_neg_{i:04d}.jpg", out)

def main():
    anchors = sorted(glob(os.path.join(ANCHOR_DIR, "*.*")))
    if not anchors:
        print("Put anchor images into", ANCHOR_DIR); return
    for a in anchors:
        name = os.path.splitext(os.path.basename(a))[0]
        print("Processing anchor:", a)
        make_dataset_per_anchor(a, name, pos_count=300, neg_count=300)
    print("Done — view dataset created.")

if __name__ == "__main__":
    main()
