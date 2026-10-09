# CNN MNIST Digit Classifier

A handwritten digit classification project built using PyTorch and a Convolutional Neural Network (CNN).

## Objective

Predict handwritten digits from 0 to 9 using 28 × 28 grayscale images from the MNIST dataset.

## Tech Stack

- Python
- PyTorch
- Torchvision
- scikit-learn

## Model Architecture

- Two convolutional layers
- ReLU activation
- Max pooling
- Fully connected layers
- Ten output classes

## Workflow

1. Load and normalize the MNIST dataset.
2. Create training, validation and test datasets.
3. Load batches using DataLoader.
4. Train the CNN using CrossEntropyLoss and Adam.
5. Select the best checkpoint using validation accuracy.
6. Evaluate the saved model on the test dataset.
7. Generate predictions for individual images.

## How to Run

Install dependencies:

`python -m pip install -r requirements.txt`

Train:

`python src/train.py`

Predict:

`python src/predict.py`

## Results

Add your actual test accuracy, test loss and classification report after running the project.

## Learning Outcomes

- CNN architecture and feature extraction
- PyTorch training and validation loops
- Model evaluation
- Saving and loading model weights
- Multiclass image classification