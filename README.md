# Cat vs. Dog Image Classification Using Support Vector Machine (SVM)

This project implements a Support Vector Machine (SVM) to classify images of cats and dogs. The objective is to build a model that accurately distinguishes between cat and dog images using supervised learning techniques.

## About the Project

Image classification is a fundamental task in machine learning and computer vision. In this project, we utilize an SVM classifier to differentiate between images of cats and dogs. The project encompasses data preprocessing, feature extraction, model training, and evaluation.

## Dataset

The dataset comprises images of cats and dogs stored in ZIP files:

- **cats.zip**: Contains images of cats.
- **dogs.zip**: Contains images of dogs.

These images are used to train and test the SVM model.

## Features

- **Data Preprocessing**: Loading images, resizing, normalization, and labeling.
- **Feature Extraction**: Extracting meaningful features from images to serve as input for the SVM.
- **Model Training**: Training an SVM classifier on the extracted features.
- **Model Evaluation**: Assessing the model's performance using appropriate metrics.

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Necessary Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `opencv-python`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/heemit/PRODIGY_ML_03.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd PRODIGY_ML_03
   ```
  
3. **Install the required packages**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn opencv-python
   ```

## Usage

1. **Extract the datasets**:   
   - Unzip cats.zip and dogs.zip into respective folders.
   
1. **Open the Jupyter Notebook:**   
   ```bash
   jupyter notebook task2.py
   ```

2. **Run the cells sequentially:**
   Execute the cells in sequence to preprocess the data, train the model, and evaluate its performance.

## Model Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and classification reports are generated to provide detailed insights into the classifier's effectiveness.
