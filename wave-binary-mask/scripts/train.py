import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

IMAGE_DIR = "./data/data/image"
MASK_DIR = "./data/data/mask-bmp"
BATCH_SIZE = 10
EPOCHS = 100
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./unet_results"
os.makedirs(SAVE_DIR, exist_ok=True)

from torch.utils.data import Dataset
import cv2, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ECGDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

        self.IMG_HEIGHT = 640
        self.IMG_WIDTH  = 1024
        MAX_SIDE = max(self.IMG_HEIGHT, self.IMG_WIDTH)

        # spatial + color augmentations
        self.train_transform = A.Compose([
            A.LongestMaxSize(max_size=MAX_SIDE),
            A.PadIfNeeded(min_height=self.IMG_HEIGHT,
                          min_width =self.IMG_WIDTH,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.ElasticTransform(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.6),
            A.Normalize(),
            ToTensorV2()
        ])

        # only spatial + normalize
        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=MAX_SIDE),
            A.PadIfNeeded(min_height=self.IMG_HEIGHT,
                          min_width =self.IMG_WIDTH,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask failed to read: {mask_path}")

        # choose transform
        if self.augment:
            aug = self.train_transform(image=image, mask=mask)
        else:
            aug = self.val_transform(image=image, mask=mask)

        image = aug['image']
        mask  = aug['mask'].unsqueeze(0).float()  # [1,H,W]

        return image, mask

images = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
masks = []

valid_images = []
for img_path in images:
    base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(MASK_DIR, base + '.bmp')
    if os.path.exists(mask_path):
        valid_images.append(img_path)
        masks.append(mask_path)

images = valid_images  # Now only keep images that have masks


# Split the dataset
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(images, masks, test_size=VAL_SPLIT + TEST_SPLIT, random_state=42)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=42)

# Dataset and DataLoaders
train_ds = ECGDataset(train_imgs, train_masks, augment=True)
val_ds   = ECGDataset(val_imgs, val_masks, augment=False)
test_ds  = ECGDataset(test_imgs, test_masks, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ------------------ MODEL ------------------
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"unet_epoch_{epoch+1}.pth"))

model.eval()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, 'loss_curve.png'))
plt.close()

def evaluate(model, loader):
    model.eval()
    iou_total = 0
    dice_total = 0
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = torch.sigmoid(model(images)) > 0.5

            intersection = (outputs & masks.bool()).float().sum((1, 2, 3))
            union = (outputs | masks.bool()).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-7)).mean().item()

            dice = (2 * intersection / (outputs.float().sum((1,2,3)) + masks.sum((1,2,3)) + 1e-7)).mean().item()

            iou_total += iou
            dice_total += dice
            count += 1

    print(f"Mean IoU: {iou_total / count:.4f}, Mean Dice: {dice_total / count:.4f}")

evaluate(model, test_loader)


def visualize_predictions(model, dataloader, num_images=5):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = torch.sigmoid(model(images)) > 0.5  # Binarize

            for i in range(images.shape[0]):
                if images_shown >= num_images:
                    return

                img = images[i].cpu().permute(1, 2, 0).numpy()
                true_mask = masks[i][0].cpu().numpy()
                pred_mask = outputs[i][0].cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img)
                axs[0].set_title('Input Image')
                axs[1].imshow(true_mask, cmap='gray')
                axs[1].set_title('Ground Truth')
                axs[2].imshow(pred_mask, cmap='gray')
                axs[2].set_title('Predicted Mask')
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()

                images_shown += 1
visualize_predictions(model, test_loader)