# Human Activity Recognition using ML principles

This repository contains code and documentation for a project on human activity recognition using machine learning principles. The goal of this project is to develop a model that can accurately classify different human activities, such as walking, running, sitting, and standing, based on sensor data from a smartphone.

## Dataset
We used the publicly available `Human Activity Recognition with Smartphones (HAR) dataset` from the `UCI Machine Learning Repository`. You can access the dataset from this [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and the repository from this [link](https://archive.ics.uci.edu/ml/index.php).

## Approach
We used Python and several machine learning libraries, including scikit-learn and TensorFlow, to develop a model for human activity recognition. Our approach involved the following steps:

1. Data preprocessing: We cleaned and normalized the data, and split it into training and testing sets.
2. Feature extraction: We extracted relevant features from the sensor data using techniques such as Principal Component Analysis (PCA) and Fast Fourier Transform (FFT).
3. Model selection: We trained and evaluated several different machine learning models, including logistic regression, support vector machines (SVM), and neural networks, to find the best model for our data.
4. Model tuning: We fine-tuned our chosen model using techniques such as hyperparameter tuning and cross-validation.
5. Evaluation: We evaluated our final model on the testing set and reported performance metrics such as accuracy, precision, recall, and F1 score.

## Usage
To run the code in this repository, you will need to install the necessary dependencies, which are listed in the requirements.txt file. You can then run the code by executing the main.py file. You can also modify the code to experiment with different machine learning models and parameters.

## Results
Our final model achieved an accuracy of 95% on the testing set, which is a promising result for human activity recognition using sensor data from a smartphone.

License
This project is licensed under the **MIT License** - see the LICENSE file for details.
