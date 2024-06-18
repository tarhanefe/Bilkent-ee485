#%% IMPORTING NECESSARY LIBRARIES

'''
Importing following libraries for the project: 

Pandas: Will be used for extracting data from the csv files and preprocessing 
    
Numpy: Will be used for linear algebra operations 
    
Matplotlib: Will be used for visualizing results
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% FUNCTION DEFINITIONS

def split(X, y, test_size=0.2, random_state=None):
    '''
    Splits a given dataset with features and labels into 2 sets with given proportions

    Parameters
    ----------
    X : np.ndarray
        2D numpy array that contains the features of the data
    y : np.ndarray
        1D numpy vector that contains the values of each data point
    test_size : float between 0 and 1 
        Proportion of the data that will be assigned to test set. The default is 0.2.
    random_state : integer, optional
        Numpy random seed. The default is None.

    Returns
    -------
    X_train : np.ndarray
        (1-test_size) proportion of the shuffled X matrix .
    X_test : np.ndarray
        test_size proportion of the shuffled X matrix .
    y_train : np.ndarray
        (1-test_size) proportion of the shuffled y vector .
    y_test : np.ndarray
        test_size proportion of the shuffled y vector.

    '''
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_size = int(n_samples * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def centralize(X):
    '''

    Centralizes the features of the matrix X by subtracting mean from each column:
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix that will be centralized

    Returns
    -------
    X_c : np.ndarray
        Centralized matrix.

    '''
    X_c = X - np.mean(X,axis = 0)
    return X_c

def standardize(X):
    '''
    Standarizes the matrix X by subtracting the mean and dividng by the standart 
    deviation of each column. Gaussian normalization. 

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    standardized_data : np.ndarray
        Standardized feature matrix.

    '''
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    standardized_data = (X - means) / stds
    return standardized_data

def normalize(X):
    '''
    Normalizes the feature matrix X by subtracting the minimum value of each 
    column and dividng by the difference between maximum and minimum value for 
    each column

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    X_normalized : np.ndarray
        Normalized feature matrix.

    '''
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    scale = np.where(max_vals - min_vals != 0, max_vals - min_vals, 1)
    X_normalized = (X - min_vals) / scale
    return X_normalized,min_vals,scale

def fit_ols(X, y):
    '''
    Finds the least squares estimate parameters for a given feature matrix and 
    values. 

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Value vector.

    Returns
    -------
    beta : np.ndarray
        Obtained OLS parameters for the given X and y.

    '''
    
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def fit_ridgels(X,y,lbd):
    '''
    This function finds the estimated parameters for a given lambda value, 
    centralized feature matrix and value vector using the ridge regression. 

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        value vector.
    lbd : float
        Regularization factor for ridge regression.

    Returns
    -------
    beta : np.ndarray
        Parameters obtained from ridge regression.

    '''
    beta = np.linalg.inv(X.T @ X + lbd*np.eye(X.shape[1])) @ X.T @ y 
    return beta

def fit_lasso(X, y, alpha=1.0, num_iterations=1000, learning_rate=0.0001, tolerance=1e-4):
    '''
    This function finds the estimated parameters for a given lambda value, 
    centralized feature matrix and value vector using the lasso regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        value vector.
    alpha : float
        Regularization factor for lasso regression. The default is 1.0.
    num_iterations : integer, optional
        Number of iterations that the regression parameters will be updated. The default is 1000.
    learning_rate : float, optional
        The learning rate for updating the values. The default is 0.0001.
    tolerance : float, optional
        Tolerance value that automatically stops the regression iterations. The default is 1e-4.

    Returns
    -------
    w : np.ndarray
        Parameters obtained from lasso regression.

    '''
    #X = X[:,1:]
    m, n = X.shape
    w = np.zeros((n, 1))
    prev_cost = float('inf')

    for i in range(num_iterations):
        # Compute predictions
        y_pred = np.dot(X, w)

        # Compute cost with L1 regularization (Lasso)
        cost = np.mean((y_pred - y) ** 2) + alpha * np.sum(np.abs(w))

        # Check for convergence
        if np.abs(prev_cost - cost) < tolerance:
            print(f"Convergence reached after {i} iterations.")
            break

        prev_cost = cost

        # Compute gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y)) + alpha * np.sign(w)
        #db = (1/m) * np.sum(y_pred - y)

        # Update weights and bias
        w -= learning_rate * dw
        #b -= learning_rate * db
    return w


def predict(X, beta):
    '''
    Finds the values corresponding to a data (feature) matrix using linear regression
    parameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    beta : np.ndarray
        Regression coefficients.

    Returns
    -------
    np.ndarray
        predicted value vector corresponding to X and y.

    '''
    return X @ beta

def mse(y_test,y_pred):
    '''
    Finds the mean squared error between the predicted and real y vectors

    Parameters
    ----------
    y_test : np.ndarray
        Vector that contains the real values.
    y_pred : np.ndarray
        Vector that contains the predicted values.

    Returns
    -------
    MSE : float
        Mean squared error between the y_test and y_pred.

    '''
    MSE = np.mean((y_test - y_pred) ** 2)
    return MSE 

def r2(y_test,y_pred):
    '''
    Coefficient of determination between y_test and y_pred vectors

    Parameters
    ----------
    y_test : np.ndarray
        Vector that contains the real values.
    y_pred : np.ndarray
        Vector that contains the predicted values.

    Returns
    -------
    R2 : float
        R2 score for the vectors y_test and y_pred.

    '''
    R2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    return R2


    
def data_process(path):
    '''
    This function gets a csv file location and converts the csv file to a feature 
    vector using feature conversions to one-hot-encoding or ordinal encoding. 

    Parameters
    ----------
    path : string
        Absolute path to the csv file.

    Returns
    -------
    X_all : np.ndarray
        All features data combined in a numpy array with an additional ones for bias parameter.
    X_all_ : np.ndarray
        All features data combined in a numpy array.
    Y_all : np.ndarray
        All values in a numpy vector.
    '''
    
    data = pd.read_csv(path)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    time_of_day_mapping = {
        'Early_Morning': 1,
        'Morning': 2,
        'Afternoon': 3,
        'Evening': 4,
        'Night': 5,
        'Late_Night': 6
    }
    data['departure_time'] = data['departure_time'].map(time_of_day_mapping)
    data['arrival_time'] = data['arrival_time'].map(time_of_day_mapping)
    
    # Convert 'stops' to integer
    stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
    data['stops'] = data['stops'].map(stops_mapping)
    
    class_mapping = {'Economy' : 0, 'Business' : 1}
    data['class'] = data['class'].map(class_mapping)
    
    # Drop the original 'flight' column
    data.drop(columns=['flight'], inplace=True)

    airline_map = {'Air_India': np.array([1,0,0,0,0,0]), 
            'AirAsia': np.array([0,1,0,0,0,0]),
            'GO_FIRST': np.array([0,0,1,0,0,0]),
            'Indigo': np.array([0,0,0,1,0,0]),
            'SpiceJet': np.array([0,0,0,0,1,0]),
            'Vistara': np.array([0,0,0,0,0,1])}
    
    airline = np.array(data['airline'].map(airline_map).to_list()).reshape((len(data),6))
    city_map = {'Bangalore': np.array([1,0,0,0,0,0]), 
            'Chennai': np.array([0,1,0,0,0,0]),
            'Delhi': np.array([0,0,1,0,0,0]),
            'Hyderabad': np.array([0,0,0,1,0,0]),
            'Mumbai': np.array([0,0,0,0,1,0]),
            'Kolkata': np.array([0,0,0,0,0,1])}
    
    source_city = np.array(data['source_city'].map(city_map).to_list()).reshape((len(data),6))
    destination_city = np.array(data['destination_city'].map(city_map).to_list()).reshape((len(data),6))
    departure_time = np.array(data['departure_time'].to_list()).reshape((len(data),1))
    stops = np.array(data['stops'].to_list()).reshape((len(data),1))
    arrival_time = np.array(data['arrival_time'].to_list()).reshape((len(data),1))
    flclass = np.array(data['class'].to_list()).reshape((len(data),1))
    duration = np.array(data['duration'].to_list()).reshape((len(data),1))
    days_left = np.array(data['days_left'].to_list()).reshape((len(data),1))
    price = np.array(data['price'].to_list()).reshape((len(data),1))
    ones = np.ones_like(flclass)
    
    X_all = np.hstack((ones,airline,source_city,destination_city,
                       departure_time,stops,arrival_time,
                       flclass,duration,days_left)).astype(float)
    
    X_all_ = X_all[:,1:]
    Y_all = price.astype(float)
   
    return X_all,X_all_,Y_all
   
def k_fold_split(X, y, k, shuffle=True, random_state=None):
    '''
    Divides the data into k subpieces that will be used for k iterations of training
    for cross validation or other purposes. 

    Parameters
    ----------
    X : np.ndarray
        Feature matrix that will be divided into subpieces after shuffling.
    y : np.ndarray
        Value vector that will be divided into subpieces after shuffling.
    k : int
        number of splits for the cross validation.
    shuffle : bool, optional
        Statement for if the data will be shuffled before training. The default is True.
    random_state : int, optional
        Random seed for initalizing the random module of numpy. The default is None.

    Returns
    -------
    folds : list
        A list that contains a test-train package for each iteration.

    '''
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    fold_sizes = X.shape[0] // k
    folds = []

    for i in range(k-1):
        test_indices = np.arange(i * fold_sizes, (i + 1) * fold_sizes)
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        folds.append((X_train, y_train, X_test, y_test))
    
    test_indices = np.arange((k-1) * fold_sizes, X.shape[0])
    train_indices = np.setdiff1d(indices, test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]    
    folds.append((X_train, y_train, X_test, y_test))
    return folds
    
def train_ols(path):
    '''
    Trains the ordinary least squares algorithm by using a document from a given
    path  using the k-folds cross validation

    Parameters
    ----------
    path : string 
        Path to the data csv.

    Returns
    -------
    mse_list : list
        A list that contains the MSE values for each iteration.
    r2_list : list
        A list that contains the R2 scores for each iteration.
    cross_val_beta : np.ndarray
        Mean of the beta values obtained through cross validation process.

    '''
    
    X_all,X_all_,Y_all = data_process(path)
    X_norm = standardize(X_all_)
    X_train, X_test, y_train, y_test = split(X_norm, Y_all,0.2,42)
    beta = fit_ols(X_train, y_train)
    test_pred = predict(X_test, beta)
    test_rmse = mse(y_test, test_pred)**0.5
    test_r2 = r2(y_test, test_pred)
    return test_rmse,test_r2

def train_ridge(path):
    '''
    Trains the ridge regression algorithm by using a document from a given
    path  using the k-folds cross validation that tries a value of lambda for each
    iteration.

    Parameters
    ----------
    path : string 
        Path to the data csv.

    Returns
    -------
    mse_list : list
        A list that contains the MSE values for each iteration.
    r2_list : list
        A list that contains the R2 scores for each iteration.
    cross_val_beta : np.ndarray
        Mean of the beta values obtained through cross validation process.
    lambdas : list
        A list that contains the tried regularization parameters.

    '''
    
    lambdas = [10.0**(i+2) for i in np.arange(-5,6,1)]
    X_all,X_all_,Y_all = data_process(path)
    rmse_list = []
    r2_list = []
    betas = []
    X_norm = standardize(X_all_)
    #Y_norm = standardize(Y_all)
    X_train, X_test, y_train, y_test = split(X_norm, Y_all,0.2,42)
    folds = k_fold_split(X_train, y_train, 10,True,42)
    cnt = 0
    for (X_train, y_train, X_val, y_val) in folds: 
        beta = fit_ridgels(X_train, y_train,lambdas[cnt])
        y_pred = predict(X_val, beta)
        dum_mse = mse(y_val, y_pred)**0.5
        dum_r2 = r2(y_val, y_pred)
        betas.append(beta)
        rmse_list.append(dum_mse)
        r2_list.append(dum_r2)
        cnt += 1
        print("%{} of the process is completed !".format(cnt/len(folds)*100))
    best_beta = betas[rmse_list.index(min(rmse_list))]
    test_pred = predict(X_test, best_beta)
    test_rmse = mse(y_test, test_pred)**0.5
    test_r2 = r2(y_test, test_pred)
    return rmse_list,r2_list,lambdas,test_rmse,test_r2
    
    
def train_lasso(path):
    '''
    Trains the lasso regression algorithm by using a document from a given
    path  using the k-folds cross validation that tries a value of lambda for each
    iteration.

    Parameters
    ----------
    path : string 
        Path to the data csv.

    Returns
    -------
    mse_list : list
        A list that contains the MSE values for each iteration.
    r2_list : list
        A list that contains the R2 scores for each iteration.
    cross_val_beta : np.ndarray
        Mean of the beta values obtained through cross validation process.
    lambdas : list
        A list that contains the tried regularization parameters.

    '''
    lambdas = [10.0**(i+2) for i in np.arange(-5,6,1)]
    X_all,X_all_,Y_all = data_process(path)
    rmse_list = []
    r2_list = []
    betas = []
    X_norm = standardize(X_all_)
    #Y_norm = standardize(Y_all)
    X_train, X_test, y_train, y_test = split(X_norm, Y_all,0.2,42)
    folds = k_fold_split(X_train, y_train, 10,True,42)
    cnt = 0
    for (X_train, y_train, X_val, y_val) in folds: 
        beta = fit_lasso(X_train, y_train,lambdas[cnt],500,learning_rate= 0.01,tolerance = 1e-4)
        y_pred = predict(X_val, beta)
        dum_mse = mse(y_val, y_pred)**0.5
        dum_r2 = r2(y_val, y_pred)
        betas.append(beta)
        rmse_list.append(dum_mse)
        r2_list.append(dum_r2)
        cnt += 1
        print("%{} of the process is completed !".format(cnt/len(folds)*100))
    best_beta = betas[rmse_list.index(min(rmse_list))]
    test_pred = predict(X_test, best_beta)
    test_rmse = mse(y_test, test_pred)**0.5
    test_r2 = r2(y_test, test_pred)
    return rmse_list,r2_list,lambdas,test_rmse,test_r2
    
#%% Training of the OLS (Ordinary Least Squares)
test_rmse,test_r2 = train_ols("/Users/efetarhan/Desktop/DATA/Clean_Dataset.csv")
print("Test RMSE of Ordinary Least Squares Estimate: {}".format(test_rmse))
print("R^2 of Ordinary Least Squares Estimate: {}".format(test_r2))



#%% Ridge Regression 

rmse_list,r2_list,lambdas,test_rmse,test_r2 = train_ridge("/Users/efetarhan/Desktop/DATA/Clean_Dataset.csv")

plt.figure(figsize = (9,4),dpi = 600)
plt.plot(lambdas[:-1],rmse_list,'-d')
plt.xlabel("Value of $\lambda$ parameter")
plt.ylabel("Root Mean Squared Error ($)")
plt.xscale("log")
plt.title("Plot of RMSE for Each Iteration for Ridge")
plt.xticks([lambdas[2*i] for i in range(len(lambdas)//2)])
plt.yticks(np.arange(21000,28000,800))
plt.show()

plt.figure(figsize = (9,4),dpi = 600)
plt.plot(lambdas[:-1],r2_list,'-d')
plt.xlabel("Value of $\lambda$ parameter")
plt.ylabel("$R^2$ Metric")
plt.xscale("log")
plt.title("Plot of $R^2$ for Each Iteration for Ridge")
plt.xticks([lambdas[2*i] for i in range(len(lambdas)//2)])
plt.yticks(np.arange(-0.6,0.15,0.07))
plt.show()

#%% Lasso Regression 

rmse_list,r2_list,lambdas,test_rmse,test_r2 = train_lasso("/Users/efetarhan/Desktop/DATA/Clean_Dataset.csv")

plt.figure(figsize = (9,4),dpi = 600)
plt.plot(lambdas[:-1],rmse_list,'-d')
plt.xlabel("Value of $\lambda$ parameter")
plt.ylabel("Root Mean Squared Error")
plt.xscale("log")
plt.title("Plot of RMSE for Each Iteration for Lasso")
plt.xticks([lambdas[2*i] for i in range(len(lambdas)//2)])
plt.yticks(np.arange(21000,38000,1500))
plt.show()

plt.figure(figsize = (9,4),dpi = 600)
plt.plot(lambdas[:-1],r2_list,'-d')
plt.xlabel("Value of $\lambda$ parameter")
plt.ylabel("$R^2$ Metric")
plt.xscale("log")
plt.title("Plot of $R^2$ for Each Iteration for Lasso")
plt.xticks([lambdas[2*i] for i in range(len(lambdas)//2)])
plt.yticks(np.arange(-2,0.15,0.2))
plt.show()
