# ECG Digitization

## Description

This project is aimed at digitizing the singals of 12 leads on an ecg paper and then predict myocardial infarction ( heart attack ) from it.

## Project Structure

  - [full-pipeline](./full-pipeline) : Code for the full pipeline
  - [notebooks](./notebooks) : Notebooks for EDA, pre-processing etc
  - [lead-segmentation](lead-segmentation) : Code to Fine-Tune YOLO v11s to detect and segment the 12 leads
  - [wave-binary-mask](wave-binary-mask) : Code to train Unet to generate binary mask from segmented lead waves.
  - [models](models) : All the weights of the trained models
   