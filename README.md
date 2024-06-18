# Flight Price Prediction

<p align="center">
  <img src="https://github.com/tarhanefe/bilkent-cs115-labs/assets/73281981/353e59fa-4cf5-4be5-b62f-afa383f3fdcd" alt="Bilkent University Logo" width = "300" />
</p>

This project is part of the EEE485 - Statistical Learning and Data Analytics course at Bilkent University. It focuses on predicting airline ticket prices using various machine learning techniques including Linear Regression, Random Forest, K-Nearest Neighbors, and Neural Networks.

## Authors

- Kemal Enes Akyüz
- Efe Tarhan

## Introduction

This project explores the relationship between airline ticket prices and various factors such as airline, destination, flight times, booking period, and flight class. The study employs custom-built machine learning models using fundamental Python libraries to predict ticket prices based on these features.

## Dataset

The dataset contains information on the prices of plane tickets from six major Indian airlines for flights between six major cities in India. The features include:

1. Airline
2. Flight
3. Source City
4. Departure Time
5. Stops
6. Arrival Time
7. Destination City
8. Class
9. Duration
10. Days Left
11. Price

## Preprocessing

Before training the models, the dataset was preprocessed:
- Non-ordered categorical data was converted to numerical values using one-hot encoding.
- The `flight` feature was removed as it did not provide significant information.
- Ordered categorical features were converted to integer values.

## Models and Methods

### Linear Regression

Implemented linear regression using ordinary least squares, ridge regression, and lasso regression.

#### Ordinary Least Squares (OLS)

This method minimizes the residual sum of squares between observed and predicted values.

#### Ridge Regression

Ridge regression adds a regularization term to the OLS to prevent overfitting.

#### Lasso Regression

Lasso regression also adds a regularization term but can shrink some coefficients to zero, effectively performing feature selection.

### K-Nearest Neighbors (KNN)

KNN predicts ticket prices based on the average price of the k-nearest points in the training set. Euclidean distance is used to find the nearest neighbors.

### Decision Trees and Random Forest

#### Decision Trees

Decision trees split the data based on feature values to create regions with minimized RSS (Residual Sum of Squares).

#### Bagging

Bootstrap Aggregation (Bagging) involves training multiple decision trees on different subsets of the training data and averaging their predictions.

#### Random Forest

Random Forests further improve bagging by considering a random subset of features for each split, reducing correlation among trees.

### Neural Networks

Neural Networks are capable of capturing non-linear relationships in the data. Different architectures were experimented with, including single-layer and multi-layer networks.

## Results

The performance of each model was evaluated using Root Mean Squared Error (RMSE) and R-squared metrics. The key findings include:
- Decision Trees and KNN showed the best performance in terms of RMSE.
- Neural Networks were competitive but required significant training time.
- Linear Regression models, including ridge and lasso, were less effective due to the non-linear nature of the data.

## How to Use

### Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/tarhanefe/bilkent-ee486.git
   cd bilkent-ee486

   ```

[Reports](https://github.com/tarhanefe/bilkent-ee485/tree/main/Reports)
[Codes](https://github.com/tarhanefe/bilkent-ee485/tree/main/Code)
