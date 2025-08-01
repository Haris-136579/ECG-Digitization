import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend first
import matplotlib.pyplot as plt  # Now import pyplot
from PIL import Image
import cv2

class WaveExtractor:
    def __init__(self, weights_path, device='cpu'):
        """
        Initialize the UNet model and set device.
        """
        print('initializing wave extractor...')
        self.model = tf.lite.Interpreter(model_path=weights_path)
        self.model.allocate_tensors()
        print('Wave extractor initialized.')

    def preprocess_for_model(self, img):
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        # Input size expected by model
        input_width = input_details[0]['shape'][3]
        input_height = input_details[0]['shape'][2]
        img_resized = img.resize((input_width, input_height))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        return img_array[np.newaxis, np.newaxis, :, :]  # shape: [1, 1, H, W]

    def extract_wave(self, cropped_lead_img):
        """
        Takes the cropped lead image in PIL Image type, returns the extracted wave mask as a numpy array.
        The output is center-cropped to remove any padding(PIL image).
        """
        # Load raw original image
        original_img = cropped_lead_img.convert("L")
        original_size = original_img.size  # (W, H)

        # Preprocess for model input
        input_tensor = self.preprocess_for_model(img=original_img)

        # Run inference
        self.model.set_tensor(self.model.get_input_details()[0]['index'], input_tensor)
        self.model.invoke()
        output_data = self.model.get_tensor(self.model.get_output_details()[0]['index'])
        pred_mask = output_data[0, 0, :, :]  # shape: (H, W)

        # Convert predicted mask to binary and resize to original size
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_mask_img = Image.fromarray(pred_mask_bin).resize(original_size, Image.BILINEAR)

        return pred_mask_img

        
    def save_wave(self, binary_mask,output_path):
        """
        saves the binary waveform mask.
        Args:
            binary_mask (PIL image): 
            output_path: duh
        """
        
        binary_mask.save(output_path)
