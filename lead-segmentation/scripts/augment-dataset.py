import os
import cv2
import random
import shutil
import numpy as np
from glob import glob

# === CONFIGURATION ===
SOURCE_IMAGE_DIR = './data/segmentation-dataset-6by2/train/images'
SOURCE_LABEL_DIR = './data/segmentation-dataset-6by2/train/labels'
DEST_IMAGE_DIR = './data/segmentation-dataset-6by2/train-augmented/images'
DEST_LABEL_DIR = './data/segmentation-dataset-6by2/train-augmented/labels'
NUM_AUGS_PER_IMAGE = 3
IMG_SIZE = (1920, 928)

os.makedirs(DEST_IMAGE_DIR, exist_ok=True)
os.makedirs(DEST_LABEL_DIR, exist_ok=True)

def apply_augmentations(image):
    # Resize to standard size first
    image = cv2.resize(image, IMG_SIZE)

    # Convert to grayscale and back to 3 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Blur
    if random.random() < 0.5:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Contrast and brightness
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return image

def apply_zoom_and_translation(image, labels, zoom_factor=1.0, tx=0, ty=0):
    h, w = image.shape[:2]
    M = np.array([
        [zoom_factor, 0, tx],
        [0, zoom_factor, ty]
    ], dtype=np.float32)

    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    updated_labels = []
    for label in labels:
        cls, x, y, bw, bh = label
        cx = x * w
        cy = y * h
        bw *= w
        bh *= h

        # Transform
        cx = zoom_factor * cx + tx
        cy = zoom_factor * cy + ty
        bw *= zoom_factor
        bh *= zoom_factor

        new_x = cx / w
        new_y = cy / h
        new_bw = bw / w
        new_bh = bh / h

        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
            updated_labels.append((cls, new_x, new_y, new_bw, new_bh))

    return image, updated_labels

def read_yolo_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x, y, bw, bh = map(float, parts[1:])
            labels.append((cls, x, y, bw, bh))
    return labels

def write_yolo_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for cls, x, y, bw, bh in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

# === AUGMENTATION LOOP ===
print("🔁 Starting augmentation...")
image_files = glob(os.path.join(SOURCE_IMAGE_DIR, '*.jpg'))

for idx, img_path in enumerate(image_files):
    base = os.path.basename(img_path).split('.')[0]
    label_path = os.path.join(SOURCE_LABEL_DIR, base + '.txt')
    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    labels = read_yolo_labels(label_path)

    for i in range(NUM_AUGS_PER_IMAGE):
        aug_img = apply_augmentations(image.copy())
        zoom = random.uniform(0.9, 1.1)
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)

        aug_img, aug_labels = apply_zoom_and_translation(aug_img, labels, zoom, tx, ty)
        if not aug_labels:
            continue

        aug_base = f"{base}_aug{i}"
        cv2.imwrite(os.path.join(DEST_IMAGE_DIR, aug_base + '.jpg'), aug_img)
        write_yolo_labels(os.path.join(DEST_LABEL_DIR, aug_base + '.txt'), aug_labels)

    if idx % 25 == 0:
        print(f"✅ Augmented {idx + 1}/{len(image_files)} images...")

print("🎉 Done. All augmentations completed successfully.")
