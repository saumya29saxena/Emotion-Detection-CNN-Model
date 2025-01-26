# Emotion-Detection-CNN-Model

This repository contains the implementation of a **Convolutional Neural Network (CNN)** for detecting human emotions from facial expressions in real-time. The project was developed as part of the **Deep Learning and Computer Vision (UCS753)** coursework.

## Project Overview

Human emotions play a crucial role in communication. This project leverages deep learning to classify emotions such as happiness, sadness, anger, and surprise using facial expression data. The model achieves high accuracy and has significant applications in mental health, human-computer interaction, and surveillance.

## Features

- **Real-Time Emotion Detection**: Classifies emotions from live video feeds or images.
- **Deep Learning Architecture**: Built using CNNs with multiple convolutional, pooling, and dropout layers.
- **Preprocessing and Augmentation**: Includes resizing, normalization, and data augmentation for robust performance.
- **Accuracy**: Achieves ~95% training accuracy and ~90% validation accuracy.

## Methodology

1. **Data Collection**: 
   - Dataset: [Kaggle Facial Expression Recognition Dataset]([https://www.kaggle.com/](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)).
   - Preprocessing: Images resized to 48x48 pixels, normalized, and augmented.
   
2. **Model Architecture**:
   - Convolutional layers to extract spatial features.
   - Dropout layers to prevent overfitting.
   - Fully connected layers for classification.
   - Softmax output for multi-class classification.

3. **Training and Evaluation**:
   - Loss Function: Categorical crossentropy.
   - Optimizer: Adam optimizer.
   - Metrics: Accuracy.
   - Results: Training accuracy ~95%, Validation accuracy ~90%.

## Results

![Training and Validation Loss](https://github.com/saumya29saxena/Emotion-Detection-CNN-Model/blob/main/Training%20and%20Validation%20Loss.png)
![Training and Validation Accuracy](https://github.com/saumya29saxena/Emotion-Detection-CNN-Model/blob/main/Training%20and%20Validation%20Accuracy.png)

- The model generalizes well across the validation dataset.
- Example prediction with a confidence score of 0.98:
  - **Detected Emotion**: Happiness.

## Applications

- **Mental Health Monitoring**: Assist therapists with real-time emotional feedback.
- **Customer Service**: Enhance user experience by adapting responses based on emotions.
- **Education**: Develop tools for emotion recognition in e-learning.
- **Security**: Strengthen surveillance systems.

## Future Work

- **Dataset Expansion**: Include diverse age groups, ethnicities, and conditions.
- **Transfer Learning**: Integrate pre-trained models like ResNet or VGG.
- **Real-Time Deployment**: Extend to IoT devices or smartphone apps.
- **Advanced Methods**: Explore attention mechanisms and ensemble techniques.
