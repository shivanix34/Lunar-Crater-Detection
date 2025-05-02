import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_data  # Ensure data_preprocessing.py is in the same folder

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Define directories
train_image_dir = 'datasets/train/images/'
train_label_dir = 'datasets/train/labels/'
valid_image_dir = 'datasets/valid/images/'
valid_label_dir = 'datasets/valid/labels/'

# Load the preprocessed training and validation data
X_train, X_val, y_train, y_val = prepare_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir)

# Display a single sample from the dataset
def display_sample(X, y, index):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(X[index])
    plt.title("Preprocessed Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(y[index].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
    plt.title("Binary Mask (Crater)")
    plt.axis('off')

    plt.show()

# Show sample data for evaluation
display_sample(X_train, y_train, index=0)  # Show the first image and mask in the training set

# Display multiple samples from the dataset
def display_multiple_samples(X, y, num_samples=5):
    plt.figure(figsize=(15, num_samples * 3))
    for i in range(num_samples):
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(X[i])
        plt.title("Image")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(y[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Binary Mask")
        plt.axis('off')

    plt.show()

# Display 5 random samples from the training set
display_multiple_samples(X_train, y_train, num_samples=5)

# Check dataset statistics
print(f"Total images in training set: {X_train.shape[0]}")
print(f"Total images in validation set: {X_val.shape[0]}")
