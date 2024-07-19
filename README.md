# Audio Deepfake Detector

This project implements an audio deepfake detection model using deep learning. The model is trained to classify audio files as either "real" or "fake" using Mel-frequency cepstral coefficients (MFCC) as features. The project includes preprocessing of audio data, building and training a neural network model, and deploying the model with Streamlit for easy use.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Audio deepfakes are artificially generated or manipulated audio samples that mimic the voice of real individuals. This project aims to detect such deepfakes using a machine learning model.

## Dataset

The dataset should be organized in a directory structure where each subdirectory corresponds to a class label, and each subdirectory contains audio files in `.wav` format.

Example structure:
