# Revolutionizing-Fake-Currency-Detection-CNN-Based-Approach-for-Indian-Rupee-Notes

This project aims to develop machine learning models capable of detecting fake Indian currency notes. The models are built using the ResNet50 architecture, fine-tuned with custom layers for binary classification to distinguish between real and fake notes for denominations of 50, 100, 200, 500, and 2000 rupees. Additionally, the EfficientNetV2B0 model is integrated for enhanced performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Augmentation](#data-augmentation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [References](#references)

## Introduction
Counterfeit currency is a significant problem in many countries, including India. This project focuses on detecting counterfeit 50, 100, 200, 500, and 2000 rupee notes using deep learning techniques. The models are trained on datasets of images of real and fake notes for each denomination and use the ResNet50 and EfficientNetV2B0 architectures for feature extraction.

## Dataset
The dataset used in this project consists of images of real and fake notes for the denominations of 50, 100, 200, 500, and 2000 rupees. The data is organized into three main directories for each denomination:
- **Training**: Contains training images of real and fake notes.
- **Validation**: Contains validation images for model evaluation during training.
- **Testing**: Contains testing images for final model evaluation.

## Model Architecture
The models are based on the ResNet50 and EfficientNetV2B0 architectures, pre-trained on the ImageNet dataset. The top layer of these models is removed, and custom layers are added for binary classification:
- **ResNet50**:
  - Flatten layer
  - Dense layer with 1024 units and ReLU activation
  - Dropout layer with a 0.5 dropout rate
  - Dense output layer with 1 unit and sigmoid activation for binary classification
- **EfficientNetV2B0**:
  - Similar custom layers as above, optimized for the EfficientNet architecture

## Data Augmentation
To improve the robustness of the models, data augmentation is applied to the training, validation, and testing datasets. The following augmentations are used:
- Rotation up to 90 degrees
- Horizontal flip
- Vertical flip

## Training
The models are compiled with binary cross-entropy loss and the Adam optimizer. Training is performed with early stopping and model checkpointing based on validation accuracy. The training process includes:
- Batch size: 8
- Epochs: 100 (with early stopping)

## Evaluation
The models' performance is evaluated using the validation and testing datasets. Metrics include accuracy, precision, recall, F1 score, number of errors, prediction time, robustness, and efficiency.

## Usage
To use the models for detecting fake notes, follow these steps:
1. Load the trained model for the specific denomination (e.g., `best_model_50.h5` for 50 rupee notes).
2. Preprocess the input image using the `preprocess_input` function.
3. Use the `predict_image` function to predict whether the note is real or fake.

Example code for prediction:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.9:
        return "Real"
    else:
        return "Fake"

test_img_path = "/path/to/image.jpg"
prediction = predict_image(finetune_model, test_img_path)
print("Prediction:", prediction)
```

## Results
The models achieved high accuracy on both the training and validation datasets for each denomination. The confusion matrix and MSE provide additional insights into the models' performance.

## Conclusion
This project demonstrates the use of deep learning for detecting fake Indian currency notes. The ResNet50 and EfficientNetV2B0-based models, combined with data augmentation, achieve high accuracy and can be used as tools to combat counterfeit currency.

## References
- ResNet50 Architecture
- EfficientNetV2B0 Architecture
- Keras Documentation
- ImageDataGenerator Documentation

Feel free to contribute to this project by raising issues or submitting pull requests. For any questions, contact Prachiborate001@gmail.com.
