# Real-Time Indian Sign Language Recognition System

## Overview

A deep learning-based Indian Sign Language Recognition System that uses MediaPipe hand tracking, Bidirectional LSTM networks, and Text-to-Speech conversion to recognize hand gestures in real time.

## Features

- Real-time webcam-based gesture recognition
- Two-hand landmark extraction
- Bidirectional LSTM sequence classification
- Text-to-Speech output using gTTS
- Confidence-based prediction filtering
- Low-latency inference

## Workflow

Video Dataset
→ Landmark Extraction
→ Numpy Sequence Generation
→ BiLSTM Training
→ Model Deployment
→ Real-Time Prediction + Audio Output

## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- MediaPipe
- cvzone
- NumPy
- gTTS

## Files

video_to_npy.py      Dataset preprocessing
train_model.py       Model training
run_cam_with_audio.py Real-time inference
model.h5             Trained model
labels.npy           Label mappings

## Installation

pip install -r requirements.txt

## Run

python run_cam_with_audio.py

## Author

R V Advaith
