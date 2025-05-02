import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score

# Function to convert YOLO annotations to a binary mask
def yolo_to_mask(label_file, img_shape):
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_center *= img_shape[1]
                y_center *= img_shape[0]
                width *= img_shape[1]
                height *= img_shape[0]
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                mask[y_min:y_max, x_min:x_max] = 1
    return mask

# Function to load and process a random image and label
def load_random_image_label(image_folder, label_folder, img_shape):
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))

    random_index = random.randint(0, len(image_files) - 1)
    image_path = os.path.join(image_folder, image_files[random_index])
    label_path = os.path.join(label_folder, label_files[random_index])

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_shape[1], img_shape[0]))

    mask = yolo_to_mask(label_path, img_shape)

    return image, mask, image_files[random_index]

# IoU function
def iou_metric(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union != 0 else 0.0

# Dice coefficient function
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# Paths to the image and label folders
image_folder = 'datasets/train/images'  # Replace with actual path
label_folder = 'datasets/train/labels'  # Replace with actual path
img_shape = (256, 256)  # Specify the desired image and mask shape (height, width)

# Load the trained model
model_path = "resnet_unet_model.h5"  # Update with your model's path
model = load_model(model_path, compile=False)

# Load a random image and label
image, label, selected_file = load_random_image_label(image_folder, label_folder, img_shape)

# Preprocess the image for the model
input_image = image.astype(np.float32) / 255.0
input_image = np.expand_dims(input_image, axis=0)
input_image = np.expand_dims(input_image, axis=-1)
input_image = np.repeat(input_image, 3, axis=-1)

# Predict the mask
predicted_mask = model.predict(input_image)
predicted_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)

# Flatten arrays for metric calculations
flattened_true = label.flatten()
flattened_pred = predicted_mask.flatten()

# Calculate metrics
precision = precision_score(flattened_true, flattened_pred)
recall = recall_score(flattened_true, flattened_pred)
f1 = f1_score(flattened_true, flattened_pred)
dice = dice_coefficient(flattened_true, flattened_pred)
iou = iou_metric(flattened_true, flattened_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(flattened_true, flattened_pred)

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(flattened_true, flattened_pred)
average_precision = average_precision_score(flattened_true, flattened_pred)

# Plot the Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Background', 'Object'])
plt.yticks([0, 1], ['Background', 'Object'])
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, color='b', label=f'AP = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Plot IoU
plt.figure(figsize=(6, 4))
plt.bar(['IoU'], [iou], color='red')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('IoU Score')
plt.show()
