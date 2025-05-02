import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model_path = "resnet_unet_model.h5"  # Update with your model's path
model = load_model(model_path, compile=False)

# Function to preprocess input image
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess the input image by resizing and normalizing.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired image size (height, width).
    Returns:
        np.ndarray: Preprocessed image.
    """
    img = cv2.imread(image_path)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    return img_normalized, original_size, img

# Function to predict the mask and postprocess the output
def predict_mask(image_path, model, target_size=(256, 256)):
    """
    Predict the crater mask for the input image.
    Args:
        image_path (str): Path to the input image.
        model (tf.keras.Model): Loaded UNet model.
        target_size (tuple): Desired image size (height, width).
    Returns:
        np.ndarray: Resized predicted mask.
    """
    img_preprocessed, original_size, original_img = preprocess_image(image_path, target_size)
    img_input = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension
    pred_mask = model.predict(img_input)[0]  # Predict mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Binarize mask
    pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))  # Resize to original size
    return pred_mask_resized, original_img

# Function to draw circles on the original image
def draw_circles(mask, original_img):
    """
    Draw circles around all detected craters.
    Args:
        mask (np.ndarray): Binary mask of detected craters.
        original_img (np.ndarray): Original input image.
    Returns:
        np.ndarray: Image with circles drawn.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = original_img.copy()
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output_img, center, radius, (0, 255, 0), 2)  # Green circle
    return output_img

# Function to overlay mask on the original image
def overlay_mask(mask, original_img):
    """
    Overlay the predicted mask on the original image.
    Args:
        mask (np.ndarray): Binary mask of detected craters.
        original_img (np.ndarray): Original input image.
    Returns:
        np.ndarray: Image with mask overlay.
    """
    overlay = original_img.copy()
    mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)  # Color the mask
    overlay = cv2.addWeighted(mask_colored, 0.5, overlay, 0.5, 0)  # Blend the mask with the original image
    return overlay


# Function to visualize input and output
def predict_and_visualize(image_path):
    """
    Predict and visualize the input image with circles and overlay.
    Args:
        image_path (str): Path to the input image.
    """
    predicted_mask, original_img = predict_mask(image_path, model)
    output_img = draw_circles(predicted_mask, original_img)
    overlay_img = overlay_mask(predicted_mask, original_img)

    # Plot original image, predicted mask, image with circles, and segmented overlay
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(predicted_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Output with Circles")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run the prediction and visualization
image_path = r"sample\6.575794217874002,7.947675057711304,62.56238320325378,63.93426404309109.png" 
predict_and_visualize(image_path)

