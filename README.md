# M-TabNet: A Transformer-Based Multi-Encoder for Early Neonatal Birth Weight Prediction Using Multimodal Data

This repository contains the code for a Transformer-based Multi-Encoder model designed to predict birth weight (BW) using multimodal data in the field of prenatal health. The model is inspired by Transformer attention mechanisms and feature selection techniques from TabNet, combined with Sparsemax for feature sparsity. This approach integrates various types of input data (e.g., nutritional, physiological, genetic, and lifestyle data) to improve predictive accuracy and model interpretability.

## Features

- **Multi-Modal Data Processing**: The model supports input from multiple modalities such as nutritional, physiological, genetic, and lifestyle data.
- **Feature Selection**: Utilizes an attention mechanism with Sparsemax to focus on the most relevant features, ensuring computational efficiency and improving model interpretability.
- **Batch Normalization**: Applied at the input layer to stabilize and accelerate the training process.
- **Model Evaluation**: Includes the calculation of performance metrics such as **Mean Absolute Error (MAE)** and **R²** for evaluating the model’s prediction accuracy.

## Requirements

The following libraries are required to run the code:

- Python 3.12
- PyTorch
- scikit-learn
- pandas
- torchmetrics
- numpy

You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
