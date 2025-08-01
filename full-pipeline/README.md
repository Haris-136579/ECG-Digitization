# Full Pipeline for Digitization of ECG Paper

This directory has the code (but not the models and data) to input a raw ecg paper(stored in data/inputs(with reference to full-pipeline directory) folder, you will have to make this folder), segment 12 leads (stored in .jpg), extract waves from the grid and digitize each wave (stored in .npy)

## Project Structure

- [configs](./configs/) : Config files (update accoridng to your need).
- [scripts](./scripts/) : Contains the following scripts (in order of execution in the pipeline) : 
  - [lead_segmentation](./scripts/lead_segmentation.py) : Crop the 12 leads using fine tuned YOLO model.
  - [lead_segmentation_onnx](./scripts/lead_segmentation_onnx.py) : Crop the 12 leads using fine tuned YOLO model in ONNX format.
  - [lead_segmentation_tflite](./scripts/lead_segmentation_tflite.py) : Crop the 12 leads using fine tuned YOLO model in the tflite format(this one will be used in the final pipeline).
  - [grid_detection](./scripts/grid_detection.py) : Get the size of 1 big square in the grid. Used for calibration later on.
  - [extract_wave](./scripts/extract_wave.py) : Extract the binary mask of wave using the 97MB UNet model.
  - [digitize](./scripts/digitize.py) : Digitize the binary mask using the big square size (according to the 97 MB UNet model).
  - [create_ecg_paper](./scripts/create_ecg_paper.py) : Reconstruct the ECG paper and label the PQRST waves
- [main](./main.py) : Main entry point 

## How to run

### Create a Virtual Environment

```
conda create -y -p ./venv python=3.10
conda activate ./venv
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run

```
python3 main.py
```

## Full Pipeline Flow:
- Input the image in [data/inputs/](./data/inputs/)
- Use the YOLO model to segment the 12 leads and save each one to [data/outputs/](./data/outputs/)
- Using those cropped images, calculate the number of pixels(per big square of the grid) to be used for calibration later on
- Use the Unet to seperate the binary wave mask(the wave with the background removed)
- Use the calculated pixel values to finally make an array from the binary masks for each wave and save them to [data/outputs/](./data/outputs/)
- Recreate the ecg paper and save it(optional). Also label the pqrs waves 

# Important Note
The unet model used in this repo is the old .pt one and not the retrained tflite one. The pipeline is final only upto the pixels calculation part. The unet inference and the rest of the pipeline is according to the old Unet model. It will need to be updated according to the new retrained tflite Unet model.