import time
start_time = time.time()

import os
import cv2
import yaml
import torch
import numpy as np

from scripts.grid_detection import get_grid_square_size
from scripts.analyze_waves import analyze_waves
from scripts.digititze import process_ecg_mask, plot_waveform
from scripts.create_ecg_paper import create_ecg_paper  # Add this import

# --- Load configs ---
with open('./configs/lead_segmentation.yaml', 'r') as f:
    lead_cfg = yaml.safe_load(f)
with open('./configs/wave_extraction.yaml', 'r') as f:
    wave_cfg = yaml.safe_load(f)
with open('./configs/grid_detection.yaml', 'r') as f:
    grid_cfg = yaml.safe_load(f)
with open('./configs/digitize.yaml', 'r') as f:
    digitize_cfg = yaml.safe_load(f)

# --- Paths ---
INPUT_IMAGE_DIR = lead_cfg['input_image_dir']
CROPPED_SAVE_DIR = lead_cfg['output_dir']
GRID_KERNEL = grid_cfg.get('closing_kernel', 10)
GRID_LENGTH_FRAC = grid_cfg.get('length_frac', 0.05)
WAVE_WEIGHTS_PATH = wave_cfg['weights_path']
WAVE_DEVICE = wave_cfg.get('device', 'cpu')
FINAL_OUTPUT_DIR = digitize_cfg['output_dir']
YOLO_WEIGHTS_PATH = lead_cfg['model_path']
os.makedirs(CROPPED_SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

if 'onnx' in YOLO_WEIGHTS_PATH.lower():
    from scripts.lead_segmentation_onnx import init_model as init_lead_model, inference_and_label_and_crop
elif 'tflite' in YOLO_WEIGHTS_PATH.lower():
    from scripts.lead_segmentation_tflite import init_model as init_lead_model, inference_and_label_and_crop
else:
    from scripts.lead_segmentation import init_model as init_lead_model, inference_and_label_and_crop

if 'tflite' in WAVE_WEIGHTS_PATH.lower():
    from scripts.extract_wave_tflite import WaveExtractor
else:
    from scripts.extract_wave import WaveExtractor

# --- 1. Lead Segmentation ---
print("Running lead segmentation...")
lead_model = init_lead_model(lead_cfg['model_path'])
image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]


all_cropped_leads = []
for img_file in image_files:
    img_path = os.path.join(INPUT_IMAGE_DIR, img_file)
    cropped_leads, _ = inference_and_label_and_crop(
        lead_model, img_path, CROPPED_SAVE_DIR, conf_threshold=lead_cfg['conf_threshold']
    )

    for crop_img, label in cropped_leads:
        base_name = os.path.splitext(img_file)[0]
        crop_path = os.path.join(CROPPED_SAVE_DIR, f"{base_name}_{label}.jpg")

        # original_size = crop_img.shape[:2]  # (height, width)

        # resized_crop = cv2.resize(crop_img, (target_width, target_height))
        cv2.imwrite(crop_path, crop_img)

        # Save original size for resizing masks later
        all_cropped_leads.append((crop_path, label, base_name, crop_path))



# --- 2. Grid Detection & Square Size Estimation ---
print("Estimating grid square sizes...")
lead_to_square_size = {}
for crop_path, label, base_name, original_size in all_cropped_leads:
    img = cv2.imread(crop_path)
    if img is None:
        print(f"Failed to read {crop_path}")
        continue
    square_size = get_grid_square_size(img, closing_kernel=GRID_KERNEL, length_frac=GRID_LENGTH_FRAC)
    print(f"Estimated square size for {base_name}_{label}: {square_size} pixels")
    lead_to_square_size[crop_path] = square_size

# --- 3. Wave Extraction (Binary Mask) ---
print("Extracting binary wave masks...")
wave_extractor = WaveExtractor(WAVE_WEIGHTS_PATH, device=WAVE_DEVICE)
lead_to_wave_mask = {}

for crop_path, label, base_name, original_size in all_cropped_leads:
    binary_mask = wave_extractor.extract_wave(crop_path)
    print(f'Extracting wave for {base_name}_{label}')
    # Resize mask back to original lead crop size (important for accurate square_size)
    # binary_mask_resized = cv2.resize(binary_mask[0][0], (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    lead_to_wave_mask[crop_path] = binary_mask
    # cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}_binarymask.jpg'), binary_mask)
    # print(f"Resized binary mask for {base_name}_{label}: {binary_mask_resized.shape}")
    # wave_extractor.plot_wave(binary_mask)


# --- 4. Digitize: Convert Mask to Waveform ---
print("Digitizing waveforms...")
lead_waveforms = []
lead_labels = []
for crop_path, label, base_name, original_size in all_cropped_leads:
    binary_mask = lead_to_wave_mask[crop_path]
    square_size = lead_to_square_size[crop_path]
    waveform = process_ecg_mask(
        binary_mask,
        square_size,
        array_path = os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}.npy'),
        # plot_path = os.path.join(FINAL_OUTPUT_DIR, f'{base_name}_{label}_waveplot.jpg')
    )
    print(f"Digitized waveform for {base_name}_{label}: length={len(waveform)}")
    lead_waveforms.append(waveform)
    lead_labels.append(label)  
    print(f"Digitized waveform for {base_name}_{label}: length={len(waveform)}")
    # plot_waveform(waveform)

# --- 5. Create ECG Paper ---
print("Creating ECG paper with all leads...")
ecg_paper_path = os.path.join(FINAL_OUTPUT_DIR, "reconstructed_ecg_paper.png")
create_ecg_paper(lead_waveforms, lead_labels, ecg_paper_path)
print("ECG paper saved to:", ecg_paper_path)

# --- 6. Analyze waves ---
analysis = analyze_waves(lead_waveforms, lead_labels)
print(f'\n\n\nFinal analysis\n\n{analysis}')

print("Pipeline complete. All outputs saved to:", FINAL_OUTPUT_DIR)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")