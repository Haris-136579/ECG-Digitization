import os
import io
import cv2
import yaml
import base64
import torch
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tempfile import NamedTemporaryFile

from scripts.grid_detection import get_grid_square_size
from scripts.analyze_waves import analyze_waves
from scripts.digititze import process_ecg_mask
from scripts.create_ecg_paper import create_ecg_paper

# --- Load configs ---
with open('./configs/lead_segmentation.yaml', 'r') as f:
    lead_cfg = yaml.safe_load(f)
with open('./configs/wave_extraction.yaml', 'r') as f:
    wave_cfg = yaml.safe_load(f)
with open('./configs/grid_detection.yaml', 'r') as f:
    grid_cfg = yaml.safe_load(f)
with open('./configs/digitize.yaml', 'r') as f:
    digitize_cfg = yaml.safe_load(f)

# --- Global Paths & Settings ---
CROPPED_SAVE_DIR = lead_cfg['output_dir']
FINAL_OUTPUT_DIR = digitize_cfg['output_dir']
YOLO_WEIGHTS_PATH = lead_cfg['model_path']
WAVE_WEIGHTS_PATH = wave_cfg['weights_path']
GRID_KERNEL = grid_cfg.get('closing_kernel', 10)
GRID_LENGTH_FRAC = grid_cfg.get('length_frac', 0.05)
WAVE_DEVICE = wave_cfg.get('device', 'cpu')

os.makedirs(CROPPED_SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# --- Model Selection ---
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

# --- Init Models ---
lead_model = init_lead_model(YOLO_WEIGHTS_PATH)
wave_extractor = WaveExtractor(WAVE_WEIGHTS_PATH, device=WAVE_DEVICE)

# --- Flask App ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/process_ecg', methods=['POST'])
def process_ecg():
    data = request.get_json()
    base64_img = data.get('image')

    if not base64_img:
        return jsonify({"error": "No image provided"}), 400

    # Save uploaded image temporarily
    img = decode_base64_image(base64_img)
    with NamedTemporaryFile(suffix=".png", delete=False) as temp:
        input_path = temp.name
        img.save(input_path)

    try:
        # --- Step 1: Lead Segmentation ---
        cropped_leads, _ = inference_and_label_and_crop(
            lead_model, input_path, CROPPED_SAVE_DIR, conf_threshold=lead_cfg['conf_threshold']
        )

        all_cropped_leads = []
        for crop_img, label in cropped_leads:
            h, w = crop_img.shape[:2]
            crop_path = os.path.join(CROPPED_SAVE_DIR, f"temp_{label}.png")
            cv2.imwrite(crop_path, crop_img)
            all_cropped_leads.append((crop_path, label, (h, w)))

        # --- Step 2: Grid Detection ---
        lead_to_square_size = {}
        for crop_path, label, (orig_h, orig_w) in all_cropped_leads:
            img = cv2.imread(crop_path)
            square_size = get_grid_square_size(img, closing_kernel=GRID_KERNEL, length_frac=GRID_LENGTH_FRAC)
            lead_to_square_size[crop_path] = square_size

        # --- Step 3: Wave Extraction + Resize to Original Size ---
        lead_to_wave_mask = {}
        for crop_path, label, (orig_h, orig_w) in all_cropped_leads:
            binary_mask = wave_extractor.extract_wave(crop_path)
            binary_mask_resized = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            lead_to_wave_mask[crop_path] = binary_mask_resized

        # --- Step 4: Digitization ---
        lead_waveforms = []
        lead_labels = []
        for crop_path, label, (orig_h, orig_w) in all_cropped_leads:
            binary_mask = lead_to_wave_mask[crop_path]
            square_size = lead_to_square_size[crop_path]
            waveform = process_ecg_mask(binary_mask, square_size)
            lead_waveforms.append(waveform)
            lead_labels.append(label)

        # --- Step 5: Create ECG Paper ---
        ecg_paper_path = os.path.join(FINAL_OUTPUT_DIR, "result_ecg_paper.png")
        create_ecg_paper(lead_waveforms, lead_labels, ecg_paper_path)

        # --- Step 6: Analyze ---
        analysis = analyze_waves(lead_waveforms, lead_labels)

        return jsonify({
            "ecg_image_base64": encode_image_to_base64(ecg_paper_path),
            "description": analysis
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
