# 🌕 Lunar Crater Detection and Segmentation

A deep learning-based solution to automatically detect and segment craters on lunar surface images using a U-Net model with a ResNet encoder.

This project is a part of a research initiative to automate crater identification on lunar surfaces. By combining U-Net's segmentation capabilities with ResNet's deep feature extraction, the model accurately highlights craters from satellite imagery.

---

## 🛠️ Tech Stack

**Programming Language**:  
- Python

**Frameworks/Libraries**:  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV, Matplotlib  
- Scikit-learn

**Model Architecture**:  
- U-Net with ResNet Backbone

---

## 📊 Model Architecture

- **Base**: U-Net  
  - U-Net is a convolutional neural network architecture specifically designed for biomedical image segmentation. It consists of an encoder-decoder structure with skip connections that help preserve spatial information during upsampling.

- **Encoder**: ResNet-34 / ResNet-50  
  - ResNet (Residual Network) introduces shortcut connections that allow gradients to flow more easily through deeper layers, improving training efficiency and accuracy. In this project, ResNet acts as the **encoder**, extracting high-level features from input images.

- **Backbone**:  
  - A *backbone* in deep learning typically refers to the core feature extractor part of a model, often based on a well-known architecture like ResNet or VGG. In our case, ResNet serves as the backbone of the U-Net, powering the encoder part of the architecture.

- **Loss Function**: Binary Cross-Entropy + Dice Loss  
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Dice Coefficient

---

## 🔍 Evaluation

Model performance is evaluated using:  
- IoU (Intersection over Union)  
- Dice Coefficient  
- Pixel Accuracy
