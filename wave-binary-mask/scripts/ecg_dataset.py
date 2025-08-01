from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2



class ECGDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

        self.IMG_HEIGHT = 640
        self.IMG_WIDTH  = 1024
        MAX_SIDE = max(self.IMG_HEIGHT, self.IMG_WIDTH)

        # For images + mask
        self.train_transform = A.Compose([
            A.LongestMaxSize(max_size=MAX_SIDE),
            A.PadIfNeeded(min_height=self.IMG_HEIGHT,
                          min_width=self.IMG_WIDTH,
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

        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=MAX_SIDE),
            A.PadIfNeeded(min_height=self.IMG_HEIGHT,
                          min_width=self.IMG_WIDTH,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
            A.Normalize(),
            ToTensorV2()
        ])

        # For images only (no mask)
        self.val_transform_nomask = A.Compose([
            A.LongestMaxSize(max_size=MAX_SIDE),
            A.PadIfNeeded(min_height=self.IMG_HEIGHT,
                          min_width=self.IMG_WIDTH,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask failed to read: {mask_path}")
            transform = self.train_transform if self.augment else self.val_transform
            aug = transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask'].unsqueeze(0).float()
            return image, mask, (original_h, original_w)
        else:
            aug = self.val_transform_nomask(image=image)
            image = aug['image']
            return image, (original_h, original_w)