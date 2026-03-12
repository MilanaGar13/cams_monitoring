import os
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw


ANCHOR_DIR = "dataset/anchor"
POS_DIR = "dataset/positive"
NEG_DIR = "dataset/negative"

os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)


def add_blur(img):
    k = random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(img, (k, k), 0)

def add_strong_blur(img):
    k = random.choice([15, 21, 31])
    return cv2.GaussianBlur(img, (k, k), 0)

def add_noise(img):
    h, w, c = img.shape
    noise = np.random.randn(h, w, c) * random.randint(5, 25)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_heavy_noise(img):
    h, w, c = img.shape
    noise = np.random.randn(h, w, c) * random.randint(40, 80)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_brightness(img):
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.int16)
    hsv[...,2] += value
    hsv[...,2] = np.clip(hsv[...,2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def add_crack(img):
    # генерируем “трещину” линиями
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for _ in range(random.randint(2, 7)):
        x1 = random.randint(0, img.shape[1])
        y1 = random.randint(0, img.shape[0])
        x2 = x1 + random.randint(-200, 200)
        y2 = y1 + random.randint(-200, 200)

        draw.line((x1, y1, x2, y2), fill=(255, 255, 255), width=random.randint(1, 3))

    return np.array(pil_img)

def uniform_image(img):
    # делаем однотонное изображение
    mean_color = np.mean(img, axis=(0,1)).astype(np.uint8)
    return np.ones_like(img) * mean_color


def process_anchor(path, count=100):
    img = cv2.imread(path)
    name = os.path.splitext(os.path.basename(path))[0]

    for i in range(count):
        # --------- POSITIVES (slightly different from anchor) ----------
        pos = img.copy()
        choice = random.choice([add_blur, add_noise, add_brightness])
        pos = choice(pos)
        cv2.imwrite(f"{POS_DIR}/{name}_pos_{i}.jpg", pos)

        # --------- NEGATIVES (strong degradation) ----------
        neg_choice = random.choice([
            add_strong_blur, add_heavy_noise, uniform_image, add_crack
        ])
        neg = neg_choice(img.copy())
        cv2.imwrite(f"{NEG_DIR}/{name}_neg_{i}.jpg", neg)


def main():
    for file in os.listdir(ANCHOR_DIR):
        if file.lower().endswith((".jpg", ".png")):
            process_anchor(os.path.join(ANCHOR_DIR, file), 150)

    print("Готово — датасет создан.")


if __name__ == "__main__":
    main()

