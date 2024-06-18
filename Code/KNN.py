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
X_all = np.hstack((standardize(airline),standardize(source_city),
                   standardize(destination_city),
                   standardize(departure_time),standardize(stops),
                   standardize(arrival_time),
                   standardize(flclass),standardize(duration),
                   standardize(days_left))).astype(float)

y_all = price.astype(float)


#%%

# Performing a test-train split
X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,0.2,42)  


# Variable used for keeping record of the cumulative rss
RSS = 0

# How many neighbors will be involved in the KNN algorithm
k = 10

# For each point in the data set
for n in range(X_test.shape[0]):

    # Predicting a result using KNN by first finding the Euclidian distances
    # between the tes point and any of the data points in the training set
    # Note that norm of the differences vector can be used for the Euclidian
    # distances

    Euclidian_distances = np.linalg.norm(X_train - X_test[n], axis=1)

    # Now, we'll find the minimums of this distances array as follows.
    # This function returns the indices of the minimum k distance values 
    minimums = np.argsort(Euclidian_distances)[0:k]
    
    # Now, the nearest k points will ve used for predicitng the price of the
    # data point by taking the mean of the k points
    y_prediction = np.mean(y_train[minimums])
    
    # Addition to the RSS for a single data point
    RSS += (y_prediction-y_test[n])**2
    
    print("{PR:.2f}%".format(PR = n/X_test.shape[0]*100))
    
# Computing the MSE for the average predictions and real price values on the 
# test data set
MSE_test = RSS/y_test.shape[0]
RMSE_n = np.sqrt(MSE_test)
print("RMSE on test data (for k = {knn}) = {r}".format(knn = k, r=RMSE_n))
    
    
