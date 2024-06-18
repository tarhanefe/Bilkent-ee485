#%% IMPORTING NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#%% FUNCTION DEFINITIONS

def standardize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    standardized_data = (X - means) / stds
    return standardized_data

def train_test_split(X, y, test_size=0.2, random_state=None):
    
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

def find_RSS(a,b):
    RSS = np.sum((a - b) ** 2)
    return RSS

def find_RSS_1(a,b):
    RSS = np.sum(abs(a - b))
    return RSS


#%% Class Definition

class SimpleNeuralNetwork:
    def __init__(self, layers, learning_rate=0.001):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights, self.biases = self.initialize_weights_he(layers)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return SimpleNeuralNetwork.sigmoid(x) * (1 - SimpleNeuralNetwork.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def identity_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def initialize_weights_he(layers):
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2 / layers[i])
            bias = np.zeros((layers[i + 1], 1))
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def feedforward(self, X):
        activations = [X.reshape(-1, 1)]
        for i in range(len(self.weights)):
            func = self.relu if i < len(self.weights) - 1 else self.identity
            activations.append(func(np.dot(self.weights[i], activations[i]) + self.biases[i]))
        return activations

    def backpropagation(self, y, activations):
        deltas = []
        y = y.reshape(-1, 1)
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                error = activations[-1] - y
                delta = error * self.identity_derivative(activations[i + 1])
            else:
                error = np.dot(self.weights[i + 1].T, deltas[-1])
                delta = error * self.relu_derivative(activations[i + 1])
            deltas.append(delta)

            weight_adjustment = np.dot(delta, activations[i].T)
            bias_adjustment = np.sum(delta, axis=1, keepdims=True)

            self.weights[i] -= self.learning_rate * weight_adjustment
            self.biases[i] -= self.learning_rate * bias_adjustment

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            total_error = 0
            for X, y in zip(X_train, y_train):
                X = X.reshape(-1, 1)
                y = y.reshape(-1, 1)

                activations = self.feedforward(X)
                self.backpropagation(y, activations)

                total_error += np.mean((y - activations[-1]) ** 2)

            avg_error = total_error / len(X_train)
            print(f"Epoch {epoch + 1}/{epochs}, Error: {avg_error}")

    @staticmethod
    def normalize_data(X):
        return X / np.max(X)


#%%

# Load the dataset
data = pd.read_csv('/Users/asus/Desktop/485P/Clean_Dataset.csv')
# Remove the 'Unnamed: 0' column
data.drop(columns=['Unnamed: 0'], inplace=True)
# Data dimensions
#print(data.shape)
# Summary of the data set
#print(data.info())

#%%
#Response

# 'price' :             The price of the flight which is a continuous variable 
#                       and target of the prediction task of this dataset


# Predictors:
    
# 'airline' :           6 categorical values that include the names of the  
#                       airline companies

# 'flight' :            The planes’ flight code which is a categorical value

# 'source_city' :       6 categorical values that denote the names of the 
#                       cities that the flight is being departed

# 'departure_time' :    6 categorical time labels indicating departure time of
#                       the flights (quantized version of the 24 hours into 
#                       6 intervals)

# 'stops' :             3 categorical values that address the number 
#                       of stops of the flight 

# 'arrival_time' :      6 categorical time labels indicating arrival time of
#                       the flights (quantized version of the 24 hours into 
#                       6 intervals)

# 'destination_city' :  6 categorical values that denote the landing location 
#                       of the flight

# 'class' :             2 categorical values which indicate if the flight is 
#                       “economy” or “business” class

# 'duration' :          A continuous feature that represents how long the 
#                       flight is expected to last

# 'days_left' :         This feature indicates how many days earlier the 
#                       ticket has been bought before the flight

#%%

# Custom mapping for 'departure_time' and 'arrival_time'
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

# Extract the numerical part of the 'flight' feature
flight_number = data['flight'].str.extract('(\d+)').astype(int)

# Find the index of the 'flight' column
flight_col_index = data.columns.get_loc('flight')

# Insert the 'flight_number' column in the place of 'flight'
data.insert(flight_col_index, 'flight_number', flight_number)

# Drop the original 'flight' column
data.drop(columns=['flight'], inplace=True)
#%%
airline_map = {'Air_India': np.array([1,0,0,0,0,0]), 
        'AirAsia': np.array([0,1,0,0,0,0]),
        'GO_FIRST': np.array([0,0,1,0,0,0]),
        'Indigo': np.array([0,0,0,1,0,0]),
        'SpiceJet': np.array([0,0,0,0,1,0]),
        'Vistara': np.array([0,0,0,0,0,1])}

airline = np.array(data['airline'].map(airline_map).to_list()).reshape((len(data),6))
#%%
city_map = {'Bangalore': np.array([1,0,0,0,0,0]), 
        'Chennai': np.array([0,1,0,0,0,0]),
        'Delhi': np.array([0,0,1,0,0,0]),
        'Hyderabad': np.array([0,0,0,1,0,0]),
        'Mumbai': np.array([0,0,0,0,1,0]),
        'Kolkata': np.array([0,0,0,0,0,1])}

source_city = np.array(data['source_city'].map(city_map).to_list()).reshape((len(data),6))
destination_city = np.array(data['destination_city'].map(city_map).to_list()).reshape((len(data),6))
#%%
departure_time = np.array(data['departure_time'].to_list()).reshape((len(data),1))
stops = np.array(data['stops'].to_list()).reshape((len(data),1))
arrival_time = np.array(data['arrival_time'].to_list()).reshape((len(data),1))
flclass = np.array(data['class'].to_list()).reshape((len(data),1))
duration = np.array(data['duration'].to_list()).reshape((len(data),1))
days_left = np.array(data['days_left'].to_list()).reshape((len(data),1))
price = np.array(data['price'].to_list()).reshape((len(data),1))
flight = np.array(data['flight_number'].to_list()).reshape((len(data),1))

# The variable used for indicating to which tree region one variable belongs 
# to (all data points are initially grouped into the region 0)
tree_regions = np.zeros_like(flclass)
#%%
X_all = np.hstack((airline,source_city,destination_city,
                   departure_time/np.max(departure_time),stops/np.max(stops),
                   arrival_time/np.max(arrival_time),
                   flclass,duration/np.max(duration),
                   days_left/np.max(days_left))).astype(float)

y_all = price.astype(float)


#%%

# Performing a test-train split
X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,0.2,42)  

# Creating the neural network
nn = SimpleNeuralNetwork(layers=[24,8,6,1], learning_rate=10**(-10))

# Seperating a validation set to be used for early stopping
X_nn_train, X_val, y_nn_train, y_val = train_test_split(X_train, y_train, 0.3, 42)

# Variable used for counting how many epoch have passed to limit the training time
esn = 0

# Variable used for stopping the training by early stopping
running = True

# Variable used for judging RMSE on the validation data
RMSE_o = 10**10

while running:

    nn.train(X_nn_train, y_nn_train, epochs = 50)
    
    RSS = 0
    for n in range(X_val.shape[0]):
        activations = nn.feedforward(X_val[n,:])
        RSS += (activations[-1]-y_val[n])**2

    # Computing the MSE for the average predictions and real price values on the 
    # validation data set
    MSE_test = RSS/y_val.shape[0]
    RMSE_n = np.sqrt(MSE_test)
    print("RMSE on validation data = {r}".format(r=RMSE_n[0][0]))

    if RMSE_n > RMSE_o:
        running = False
        print("Ended training due to early stopping")
        
    RMSE_o = RMSE_n
    
    esn += 1
    
    if esn == 200:
        running = False
        print("Ended training due to reaching 10000 epochs")
        
        
RSS = 0
for n in range(X_test.shape[0]):
    activations = nn.feedforward(X_test[n,:])
    RSS += (activations[-1]-y_test[n])**2

# Computing the MSE for the average predictions and real price values on the 
# test data set
MSE_test = RSS/y_test.shape[0]
RMSE_n = np.sqrt(MSE_test)
print("RMSE on test data = {r}".format(r=RMSE_n[0][0]))
    
