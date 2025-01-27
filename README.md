# Handwritten Character Recognition using ANN and CNN

This repository contains implementations of handwritten character recognition algorithms using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The project explores the capabilities, challenges, and performance metrics of these models, comparing their accuracy, loss, and classification reports.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Implementation](#implementation)
  - [Artificial Neural Networks (ANN)](#artificial-neural-networks-ann)
  - [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
- [Performance Metrics](#performance-metrics)
- [Visualization](#visualization)
- [Challenges](#challenges)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [License](#license)

---

## Introduction

Handwritten character recognition is a critical application of deep learning, enabling advancements in Optical Character Recognition (OCR) systems. This project involves designing, training, and evaluating models to classify handwritten digits with high accuracy using:
1. **Artificial Neural Networks (ANN)**
2. **Convolutional Neural Networks (CNN)**

The project also provides a detailed comparison of performance metrics, highlighting the strengths and challenges of each approach.

---

## Dataset

### Source
- The models were trained on the MNIST dataset and a Kannada digit dataset.
- Each image is a 28x28 grayscale image representing a single digit.

### Preprocessing
- **Normalization**: Pixel values were normalized to the range [0, 1].
- **Reshaping**: Data was reshaped to fit the input requirements of ANN and CNN models.

---

## Implementation

### Artificial Neural Networks (ANN)
- Input: Flattened 28x28 images into 784-element vectors.
- Architecture:
  - Fully connected layers with `ReLU` activation.
  - Output layer with `Softmax` for multiclass classification.
- Loss Function: Sparse Categorical Crossentropy.
- Optimizer: Adam.
- Metrics: Accuracy.

### Convolutional Neural Networks (CNN)
- Input: 28x28 images with a single channel.
- Architecture:
  - Convolutional layers with `ReLU` activation.
  - MaxPooling layers for dimensionality reduction.
  - Fully connected layers.
  - Output layer with `Softmax`.
- Loss Function: Sparse Categorical Crossentropy.
- Optimizer: Adam.
- Metrics: Accuracy.

---

## Performance Metrics

### Key Metrics
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Correctly predicted positive observations / total predicted positives.
- **Recall (Sensitivity)**: Correctly predicted positive observations / actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

### Observations
- **CNN** achieved a higher accuracy (~99.23%) compared to ANN.
- **ANN** demonstrated good performance but struggled with complex patterns due to lack of spatial feature extraction.

### Classification Reports
- **Macro Average**: Consistent performance across all classes.
- **Weighted Average**: Adjusted performance considering class distribution.

---

## Visualization

### Training Curves
Graphs showcasing the improvement in accuracy and reduction in loss over epochs for both ANN and CNN models.

### Classification Report Insights
- High precision and recall values across most classes.
- CNN outperformed ANN in handling complex patterns.

---

## Challenges

### Difficulties with ANN
- Lack of spatial feature extraction limited its ability to recognize complex patterns.
- Higher error rates for visually similar digits (e.g., 3 and 8).

### General Challenges
- Balancing precision and recall for imbalanced datasets.
- Overfitting with deeper architectures.

---

## Conclusion

- **CNN** is a superior choice for handwritten character recognition due to its ability to extract spatial features, resulting in higher accuracy.
- While **ANN** is simpler and faster, it is less effective for this task.

---

