import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Define the paths to the dataset
train_images_path = '/kaggle/input/crater-data/LU3M6TGT_yolo_format/train/images'
valid_images_path = '/kaggle/input/crater-data/LU3M6TGT_yolo_format/valid/images'

# Create an instance of ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Load images from the directory
train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/crater-data/LU3M6TGT_yolo_format/train/',  # This will use the images from the images subdirectory
    target_size=(224, 224),  # ResNet50 expects 224x224 images
    batch_size=32,
    class_mode='binary'  # Since it's crater detection (binary classification)
)

validation_generator = validation_datagen.flow_from_directory(
    '/kaggle/input/crater-data/LU3M6TGT_yolo_format/valid/',  # This will use the images from the images subdirectory
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load the ResNet50 model with pre-trained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Add L2 regularization
    Dropout(0.2),  # Add dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
model.fit(
    train_generator,
    steps_per_epoch=100,
    validation_data=validation_generator,
    validation_steps=50,
    epochs=20,  # Increase epochs to allow for early stopping
    callbacks=[early_stopping]  # Add early stopping
)
