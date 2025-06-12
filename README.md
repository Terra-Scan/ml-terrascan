# TerraScan Machine Learning Project: Image Classification with TensorFlowAdd commentMore actions

Welcome to the TerraScan Machine Learning repository! This project focuses on classifying images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project leverages Streamlit for the user interface, allowing easy interaction with the model through a web app.

## Table of Contents

- [Project Description](#project-description)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [How the Model Works](#how-the-model-works)
  - [Model Architecture](#model-architecture)
  - [Training the Model](#training-the-model)
- [Files](#files)

## Project Description

The **TerraScan Machine Learning** project is a deep learning model for classifying images using a CNN. The model is trained on a dataset of images, and a simple user interface is built using **Streamlit** to interact with the trained model.

This repository contains:

- Code to preprocess image data.
- A CNN model built with TensorFlow/Keras.
- A Streamlit web app to run predictions.
- Instructions on setting up and running the project locally.

## Setup and Installation

1. **Clone the Repository**:

   ```bash
    git clone https://github.com/Terra-Scan/machine-learning.git
    cd machine-learning
   ```

2. **Create a Virtual Environment and Activate It**:

   ```bash
    pipenv install
    pipenv shell
   ```

3. **Install the Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Running the App**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```bash
machine-learning/
│
├── app.py                 # The Streamlit app for interacting with the model.
├── labels.txt             # A text file containing the class labels for predictions.
├── mapping_info.py        # Script to map image data to labels and class information.
├── model_terrascan.h5     # The trained Keras model (saved after training).
├── requirements.txt       # List of Python libraries needed to run the project.
├── split_dataset.py       # Script to split the dataset into training and testing.
├── train_model.ipynb      # Jupyter notebook for training the model.
│
├── dataset/               # Folder containing the original image dataset.
│
└── dataset_split/         # Folder containing the split dataset (train/test).
```

## How the Model Works

### Model Architecture

The model is based on a Convolutional Neural Network (CNN) and consists of several layers:

1. Convolutional Layers: To extract features from input images.
2. MaxPooling Layers: To reduce the dimensionality of feature maps.
3. Dense Layers: Fully connected layers to make predictions based on extracted features.

Activation Functions: Using ReLU for intermediate layers and a softmax or sigmoid activation in the output layer for classification.

### Training the Model

The model is trained on a dataset of images. We use TensorFlow's Keras API to define and compile the model, followed by training the model with the following steps:

1. Data Preprocessing: Images are resized and augmented to improve the model's generalization.
2. Training: The model is trained using backpropagation and a suitable optimizer (such as Adam).
3. Evaluation: After training, the model is validated on a separate dataset to check its accuracy.

## Files

- `app.py`: The main Streamlit app where users can upload images and view predictions.
- `dataset/`: Directory that holds the image dataset for training and validation.
- `dataset_split/`: Folder containing the split dataset with training and testing subfolders for image data.
- `split_dataset.py`: Python script to split the dataset into training and validation sets.
- `requirements.txt`: List of Python libraries needed to run the project.
- `model_terrascan.h5`: The saved Keras model after training.
- `labels.txt`: A file containing the class labels.
- `mapping_info.py`: A Python script for mapping image data to class labels for training and predictions.
- `train_model.ipynb`: Jupyter notebook containing the code for training, evaluating, and saving the model.
