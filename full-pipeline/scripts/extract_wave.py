import torch
import cv2
from segmentation_models_pytorch import Unet
import matplotlib
matplotlib.use('Agg')  # Set the backend first
import matplotlib.pyplot as plt  # Now import pyplot
from scripts.ecg_dataset import ECGDataset

class WaveExtractor:
    def __init__(self, weights_path, device='cpu'):
        """
        Initialize the UNet model and set device.
        """
        print('initializing wave extractor...')
        self.device = torch.device(device)
        self.model = Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print('Wave extractor initialized.')

    @staticmethod
    def center_crop(tensor, target_h):
        # tensor shape: [C, H, W]
        _, h, w = tensor.shape
        top = (h - target_h) // 2
        return tensor[:, top:top+target_h, :]

    def extract_wave(self, cropped_lead_img_path):
        """
        Takes the path to a cropped lead image, returns the extracted wave mask as a numpy array.
        The output is center-cropped to remove any padding.
        """
        # Use ECGDataset for preprocessing
        ds = ECGDataset([cropped_lead_img_path], mask_paths=None, augment=False)
        img_tensor, (orig_h, orig_w) = ds[0]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            output = torch.sigmoid(self.model(img_tensor))
            mask = (output > 0.5).float().cpu()[0, 0]  # shape: [H, W]
            # Center crop to remove padding
            mask_cropped = self.center_crop(mask.unsqueeze(0), orig_h)[0]
            binary_mask = (mask_cropped > 0.5).cpu().numpy().astype('uint8')
        return binary_mask
    
    def plot_wave(self, binary_mask, title="Extracted Waveform Binary Mask"):
        """
        Plots the binary waveform mask.
        Args:
            binary_mask (np.ndarray): 2D binary mask (output of extract_wave)
            title (str): Plot title
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(binary_mask, cmap='gray', aspect='auto')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
