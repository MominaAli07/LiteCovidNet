# LiteCovidNet: Lightweight CNN for Lung X-Ray COVID-19 Diagnosis

This repository contains the code for **LiteCovidNet**, a lightweight convolutional neural network framework for multi-class classification of lung diseases (including COVID-19) from chest X-ray images. The project also features robust uncertainty estimation and an interactive [Streamlit](https://streamlit.io/) web app for rapid diagnosis.

## Features

- **Custom lightweight CNN architecture** trained from scratch (no transfer learning)
- Supports multi-class classification: COVID-19, Lung Opacity, Normal, Pneumonia, and TB
- Uncertainty quantification using Stochastic Gradient Langevin Dynamics (SGLD)
- Predictive entropy, variation ratio, and calibration error computation
- Streamlit web interface for real-time image upload and prediction
- Extensive model evaluation and visualization tools

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/liteCovidNet.git
cd liteCovidNet
```
---
### 2. Check the model code
```bash
python custom.py
```
---
### 3. To cite this work:
```bash
@misc{your_litecovidnet_2025,
  author = {Momina L. Ali},
  title = {LiteCovidNet: Lightweight CNN for COVID-19 Diagnosis and Uncertainty Estimation},
  year = {2025},
  url = {https://github.com/yourusername/litecovidnet}
}
```
