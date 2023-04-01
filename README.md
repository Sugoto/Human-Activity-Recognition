# Human Activity Recognition using Deep Learning 🏃‍♂️🏃‍♀️💻🧠
This repository contains code and documentation for a project on human activity recognition using machine learning principles. The goal of this project is to develop a model that can accurately classify different human activities, such as walking 🚶‍♂️, running 🏃‍♀️, sitting 🪑, and standing 🕴️, based on sensor data from a smartphone 📱.

## Dataset 📊📈
We used the publicly available Human Activity Recognition with Smartphones (HAR) dataset from the `UCI Machine Learning Repository`. You can access the dataset from this [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and the repository from this [link](https://archive.ics.uci.edu/ml/index.php).

## Approach 1: Simple ML models 🧬🔍🤖
This implementation employs a basic approach to machine learning for recognizing human activities. The process comprises several stages, including data preprocessing, model selection, model evaluation, and results summary.

1. Data preprocessing: involves loading data from text files, splitting it into training and testing sets, and performing some necessary manipulations.

2. Model selection: is achieved through defining a collection of standard models, including decision trees, support vector machines, K-nearest neighbors, and ensemble models such as gradient boosting, extra trees, bagging, and random forests.

3. Model evaluation: is carried out by training and assessing individual models with the training and testing data. The resulting accuracy score is returned. The evaluate_models function evaluates a dictionary of models, returning a dictionary of accuracy scores for each model.

## Approach 2: Using LSTM model with Convolution 🧬🔍🤖
This implementataion used a deep learning approach for classification using a Convolutional Long Short-Term Memory (LSTM) neural network. It involves the following steps:

1. Data preprocessing: The training and testing data is passed to the evaluate_model function as trainX, trainy, testX, and testy.

2. Model definition: The LSTM model is defined using the Sequential API from the Keras library. It includes a ConvLSTM2D layer with 64 filters, a kernel size of (1, 3), and ReLU activation function. This is followed by a Dropout layer to prevent overfitting, a Flatten layer to convert the 5D output from the ConvLSTM2D layer to a 2D input for the Dense layer, and two Dense layers with 100 and n_outputs neurons, respectively. The final layer uses a softmax activation function to output class probabilities.

3. Model compilation: The model is compiled using the categorical_crossentropy loss function, the Adam optimizer, and accuracy as the evaluation metric.

4. Model training: The model is trained on the training data for a specified number of epochs and batch size using the fit function. The validation data is used to monitor the loss and accuracy during training. An early stopping callback is used to stop training if the validation loss does not improve after a specified number of epochs.

5. Model evaluation: The accuracy of the trained model is evaluated using the test data and the evaluate function.

## Usage 💻🚀
The notebook provided in the repository contains code for performing human activity recognition (HAR) classification using machine learning. It includes code for loading and preprocessing the dataset, as well as implementing and evaluating various machine learning models for classification. The notebook can be executed in a Colab environment, and allows for modification and experimentation with different models and parameters.

## Results 📈🔍
Our final model achieved an accuracy of 95% on the testing set, which is a promising result for human activity recognition using sensor data from a smartphone.

## License 📝
This project is licensed under the MIT License - see the LICENSE file for details.
