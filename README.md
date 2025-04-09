
# M-TabNet: A Transformer-Based Multi-Encoder for Early Neonatal Birth Weight Prediction Using Multimodal Data

This repository contains the implementation of a **Transformer-based Multi-Encoder** model for predicting **birth weight (BW)** using **multimodal data** in the field of **prenatal health**. The model leverages **Transformer**-based attention mechanisms and **TabNet**'s feature selection techniques, combined with **Sparsemax** for enforcing sparsity and improving model interpretability.

The model integrates multiple types of data, including **nutritional**, **physiological**, **genetic**, and **lifestyle** data to predict **birth weight (BW)**.

## Features

- **Multimodal Data Handling**: The model handles multiple data types (nutritional, physiological, genetic, and lifestyle).
- **Sparse Feature Selection**: Uses **Sparsemax** to enforce sparsity in the feature selection process.
- **Batch Normalization**: Applied at the input layer to ensure stable and efficient training.
- **Model Evaluation**: Tracks performance using metrics like **Mean Absolute Error (MAE)** and **R²**.

## Directory Structure

The directory structure of the repository is as follows:

```
Transformer-Multi-Encoder-Prenatal-Health/
│
├── data_loader.py            # Logic for loading and processing the data from Excel files
├── dataset.py                # Custom Dataset class to handle features and labels
├── feature_transformer.py    # Defines the Feature Transformer (GLU-based)
├── sparsemax.py              # Sparsemax implementation for sparse feature selection
├── tabnet_encoder.py         # TabNet Encoder class that handles attention mechanism
├── tabnet.py                 # Main TabNet model that integrates all encoders
├── train.py                  # Script for training the model, setting up optimizer, and loss
├── main.py                   # Main script to run and execute the model (entry point)
├── nutritionalgit.xlsx       # Excel file with nutritional data
├── physiologicalgit.xlsx     # Excel file with physiological data
├── genetics.xlsx             # Excel file with genetic data
├── lifestylegit.xlsx         # Excel file with lifestyle data
├── target.xlsx               # Excel file containing the target variable (BW)
├── requirements.txt          # List of required dependencies for setting up the environment
└── .gitignore                # Specifies files to be ignored in the GitHub repository
```

## Requirements

This project requires Python 3.x and several libraries. You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

The dependencies are:

- **torch**: The core deep learning library used for building and training the model.
- **pandas**: A data manipulation library, used for processing Excel files and handling the datasets.
- **scikit-learn**: For model evaluation, particularly for calculating MAE and R².
- **torchmetrics**: Provides additional metrics for evaluating PyTorch models.
- **openpyxl**: Required for reading Excel files.

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Transformer-Multi-Encoder-Prenatal-Health.git
cd Transformer-Multi-Encoder-Prenatal-Health
```

### 2. Install Dependencies

Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Make sure the following Excel files are available in your project directory:

- **nutritionalgit.xlsx**: Nutritional data
- **physiologicalgit.xlsx**: Physiological data
- **genetics.xlsx**: Genetic data
- **lifestylegit.xlsx**: Lifestyle data
- **target.xlsx**: Target variable (BW)

### 4. Running the Model

To run the model, execute the `main.py` script:

```bash
python main.py
```

This will load the data, train the model, and output performance metrics (MAE and R²) for each epoch.

### 5. Customization

You can modify the file paths, model parameters, or data processing logic in the code. Just ensure the data files are correctly formatted.

## Citation

If you use this code in your research, please cite this repository as follows:

```
@misc{your-repository,
  author = {Your Name},
  title = {Transformer-based Multi-Encoder Model for Prenatal Health Prediction},
  year = {2025},
  url = {https://github.com/your-username/Transformer-Multi-Encoder-Prenatal-Health},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by the **TabNet** model for sequential decision-making using attention mechanisms.
- Thanks to the research community for advancing multimodal machine learning techniques.
