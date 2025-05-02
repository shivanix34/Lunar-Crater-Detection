import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from data_preprocessing import yolo_to_mask, load_yolo_dataset

# Set image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

def resnet_block(inputs, filters):
    # If the input channels are not equal to filters, apply a convolution to match the dimensions
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, (1, 1), padding='same')(inputs)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, inputs])  # Skip connection
    x = layers.ReLU()(x)
    return x

# Create the ResNet-UNet model
def create_resnet_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = resnet_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = resnet_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = resnet_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = resnet_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    bottleneck = resnet_block(p4, 1024)

    # Decoder
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u4 = layers.concatenate([u4, c4])
    c5 = resnet_block(u4, 512)

    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u5 = layers.concatenate([u5, c3])
    c6 = resnet_block(u5, 256)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u6 = layers.concatenate([u6, c2])
    c7 = resnet_block(u6, 128)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u7 = layers.concatenate([u7, c1])
    c8 = resnet_block(u7, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c8)  # Use 'softmax' for multi-class

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Prepare data for training
def prepare_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, valid_size=0.2, train_fraction=0.1):
    X_train, y_train = load_yolo_dataset(train_image_dir, train_label_dir)

    # Sample a fraction of the training data
    num_samples = int(len(X_train) * train_fraction)
    X_train, y_train = X_train[:num_samples], y_train[:num_samples]

    if valid_image_dir and valid_label_dir:
        X_val, y_val = load_yolo_dataset(valid_image_dir, valid_label_dir)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)

    return X_train, X_val, y_train, y_val

# Main execution
if __name__ == "__main__":
    # Define directories
    train_image_dir = 'datasets/train/images/'
    train_label_dir = 'datasets/train/labels/'
    valid_image_dir = 'datasets/valid/images/'
    valid_label_dir = 'datasets/valid/labels/'

    # Prepare the data
    X_train, X_val, y_train, y_val = prepare_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir)

    # Create and compile the model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = create_resnet_unet(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Set the number of epochs and batch size
    epochs = 10
    batch_size = 50  # You can also reduce the batch size to speed up training further

    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=epochs, 
                        batch_size=batch_size)

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Make predictions on a sample image
    import matplotlib.pyplot as plt

    sample_image = X_val[0:1]  # Take a single image for prediction
    predicted_mask = model.predict(sample_image)

    # Visualize the result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(sample_image[0])

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(y_val[0].squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask[0].squeeze(), cmap='gray')
    plt.show()
