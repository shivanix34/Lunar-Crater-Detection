import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Function to convert YOLO format to binary mask
def yolo_to_mask(label_file, img_shape):
    """
    Converts YOLO bounding box annotations to a binary mask.
    
    Arguments:
    - label_file: Path to the YOLO label file (.txt)
    - img_shape: Shape of the corresponding image (height, width)

    Returns:
    - mask: Binary mask with craters marked as 1 and background as 0
    """
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # YOLO format has normalized coordinates (0 to 1), scale them to image size
                x_center *= img_shape[1]  # Convert to pixel coordinates
                y_center *= img_shape[0]
                width *= img_shape[1]
                height *= img_shape[0]
                
                # Calculate the bounding box edges
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                # Set the corresponding region in the mask to 1 (crater region)
                mask[y_min:y_max, x_min:x_max] = 1
                
    return mask

# Load images and convert YOLO labels to masks
def load_yolo_dataset(image_dir, label_dir):
    """
    Loads images and corresponding YOLO labels and converts the labels into binary masks.
    
    Arguments:
    - image_dir: Path to the directory containing images
    - label_dir: Path to the directory containing YOLO label files (.txt)

    Returns:
    - images: List of preprocessed images
    - masks: List of corresponding binary masks
    """
    images = []
    masks = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize image
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            
            # Load corresponding YOLO label and convert to mask
            label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(label_dir, label_filename)
            mask = yolo_to_mask(label_path, (IMG_HEIGHT, IMG_WIDTH))
            masks.append(mask)
    
    images = np.array(images)
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)  # Add channel dimension to masks

    return images, masks

# Function to prepare data for training
def prepare_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, test_size=0.2):
    """
    Prepares training and validation datasets, splitting them if needed.
    
    Arguments:
    - train_image_dir: Path to training images directory
    - train_label_dir: Path to training labels directory
    - valid_image_dir: Path to validation images directory
    - valid_label_dir: Path to validation labels directory
    - test_size: Fraction of the training data to be used as validation set

    Returns:
    - X_train, X_val, y_train, y_val: Arrays of training and validation images and masks
    """
    # Load training data
    X_train, y_train = load_yolo_dataset(train_image_dir, train_label_dir)
    
    # If a separate validation set is provided, use it; otherwise split the training set
    if valid_image_dir and valid_label_dir:
        X_val, y_val = load_yolo_dataset(valid_image_dir, valid_label_dir)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, X_val, y_train, y_val

# Example usage
if __name__ == "__main__":
    # Define directories
    train_image_dir = 'datasets/train/images/'
    train_label_dir = 'datasets/train/labels/'
    valid_image_dir = 'datasets/valid/images/'
    valid_label_dir = 'datasets/valid/labels/'

    # Prepare the data
    X_train, X_val, y_train, y_val = prepare_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir)
    
    # Output the shapes of the prepared data
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
