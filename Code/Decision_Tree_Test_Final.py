# -*- coding: utf-8 -*-


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
    
def k_fold_split(X, y, k, shuffle=True, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_temp = X[indices]
        y_temp = y[indices]

    fold_sizes = X.shape[0] // k
    folds = []

    for i in range(k-1):
        test_indices = np.arange(i * fold_sizes, (i + 1) * fold_sizes)
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, y_train = X_temp[train_indices], y_temp[train_indices]
        X_test, y_test = X_temp[test_indices], y_temp[test_indices]

        folds.append((X_train, y_train, X_test, y_test))
    
    test_indices = np.arange((k-1) * fold_sizes, X.shape[0])
    train_indices = np.setdiff1d(indices, test_indices)

    X_train, y_train = X_temp[train_indices], y_temp[train_indices]
    X_test, y_test = X_temp[test_indices], y_temp[test_indices]
    
    folds.append((X_train, y_train, X_test, y_test))
    return folds

def find_RSS(a,b):
    RSS = np.sum((a - b) ** 2)
    return RSS

def find_RSS_1(a,b):
    RSS = np.sum(abs(a - b))
    return RSS

def find_R_values(X,y):
    # Creating an array that'll contain the mean for every region created by 
    # the tree algorithm, this array is used for estimating values for new data
    # (The size of the array is the same as the number of regions in the tree)
    R_values = np.zeros(int(np.max(X[:,-1]))+1)
    
    # For each region in the tree (region number is in -1th column)
    for i in range (0,int(np.max(X[:,-1]))+1):
        # Finding the points that are all in the region i
        R_indices = np.argwhere(X[:,-1]==i)
        # Noting the means of the points on the region i
        R_values[i] = np.mean(y[R_indices])
    return R_values

def choose_predictor(X_reg,y_reg,Dec_Bou):
    # This function chooses among which of the predictors will be the best to 
    # split the data from and which value is the best decision boundary
    
    # This array will hold the minimum RSS value [0] for each predictor and the
    # decision boundary it denotes [1]
    RSS_pre = np.zeros((X_reg.shape[1]-1,2))
    # For each predictor (note that last column is used for tree regions)
    for pre in range(0,X_reg.shape[1]-1):
        # This variable will hold the minimum RSS value for the predictor 
        min_RSS = 10**50
        # This variable will denote the decision boundary for minimum RSS
        min_val = 0
        for bou in Dec_Bou[pre]:
            # Finding the data points under the decision boundary
            down_reg = np.squeeze(y_reg[np.argwhere(X_reg[:,pre]<bou)])
            # Finding RSS for the points under the decision boundary
            RSS_down = np.linalg.norm(down_reg-np.mean(down_reg))**2
            # Finding the data points over the decision boundary
            up_reg = np.squeeze(y_reg[np.argwhere(X_reg[:,pre]>=bou)])
            # Finding RSS for the points over the decision boundary
            RSS_up = np.linalg.norm(up_reg-np.mean(up_reg))**2
            if ((RSS_down + RSS_up)<min_RSS):
                min_RSS = RSS_down + RSS_up
                min_val = bou
        # This is the best decision boundary for the given predictor        
        RSS_pre[pre,0] = min_RSS
        RSS_pre[pre,1] = min_val
    
    cho_pre = np.argmin(RSS_pre[:,0])
    pre_value = RSS_pre[cho_pre,1]
    
    return cho_pre, pre_value 
        
def create_decision_boundaries(X_sorted):
    # This function creates possible decision boundaries based on the sorted 
    # data set
    
    # Creating the dictionary that will contain the outputs
    Dec_Bou = dict.fromkeys((range(X_sorted.shape[1]-1)),[])
    
    # For each predictor, finding the unique values for that predictor
    for pre in range(0,X_sorted.shape[1]-1):
        Dec_Bou[pre]=(np.unique(X_sorted[:,pre]))

    return Dec_Bou

def Decision_Tree_Constructer(X_train,y_train,max_reg,min_points):
    # max_reg : maximum number of regions to be created by the tree branches
    # min_points : number of points after which a region will not be seperated
    X = X_train
    y = y_train
    
    # Creating the dictionary that will contain all the node relations by 
    # summining the next two nodes to follow if it is not a terminal node and 
    # the number of the region that it flows to if it is a terminal node
    Tree_Nodes = {0: [True,0]}
    # For terminal nodes:
    # {i: [Is end node = True, number of the region it denotes]}
    # For internal nodes:
    # {j : [Is terminal node = False, predictor it splits, value of split, 
    #       node 1 (precitor < value), node 2 (precitor >= value)]}
    
    # Creating an array contaning all the values of the predictors in ascending 
    # order (this array will be useful for the tree algorithm where the jumbled up
    # data order does not matter)
    X_sorted = np.zeros_like(X)
    for pre in range(0,X.shape[1]-1):
        X_sorted[:,pre] = np.sort(X[:,pre])
        
    # Creating the array that will contain the would be decision boundaries
    # for each predictor    
    Dec_Bou = create_decision_boundaries(X_sorted)
    
    # The variable used for keeping track of number of regions
    num_reg = 0
    # The variable used for keeping track of number of nodes
    num_nodes = 0
    
    while(True):
        
        # For each node in the tree
        for node in list(Tree_Nodes):
            # Checking whether the max region limit is reached
            if num_reg >= max_reg:
                # If maximum region number is reach the tree construction ends
                break
            # If it is a terminal node
            if (Tree_Nodes[node][0] == True):
                # Region associated with the chosen node
                node_reg =  Tree_Nodes[node][1] 
                # Getting all the data in the terminal node region
                X_reg = np.squeeze(X[np.argwhere(X[:,-1]==node_reg),:])
                y_reg = np.squeeze(y[np.argwhere(X[:,-1]==node_reg)])
                # If there are already less than the minimum amount of points 
                # in a region, it will not be further seperated
                if (len(X_reg.shape)==2):
                    # Choosing a predictor to split and where to split it
                    cho_pre, pre_value = choose_predictor(X_reg,y_reg,Dec_Bou)
                    # Creating two new terminal nodes for the split
                    Tree_Nodes[num_nodes+1] = [True,node_reg]
                    X[np.argwhere((X[:,cho_pre]<pre_value) & (X[:,-1]==node_reg)),-1] = node_reg
                    Tree_Nodes[num_nodes+2] = [True,num_reg+1]
                    X[np.argwhere((X[:,cho_pre]>=pre_value) & (X[:,-1]==node_reg)),-1] = num_reg+1
                    # Altering the original node because it is no longer terminal
                    Tree_Nodes[node] = [False,cho_pre,pre_value,num_nodes+1,num_nodes+2]
                    num_nodes += 2
                    num_reg += 1
        
        if num_reg >= max_reg:
            break
    return Tree_Nodes, X

def predict(X,Tree_Nodes,R_values):
    # This function predicts the price values for a given preddictor set based
    # on the region in which it falls on the tree structure and the mean at 
    # that region
    
    # Temporary variable used for keeping the nodes to which points are diverted
    X_nodes = np.zeros(X.shape[0])
    
    # The variable that'll hold the predicted value of price
    y_val_pre = np.zeros(X.shape[0])
    
    # For each node
    for node in list(Tree_Nodes):
        # If the node is not a terminal node, the points are sent to the
        # following nodes through the structure of the tree
        if Tree_Nodes[node][0] == False:
            X_nodes[np.argwhere((X_nodes[:] == node) & (X[:,Tree_Nodes[node][1]]<Tree_Nodes[node][2]))] = Tree_Nodes[node][3]
            X_nodes[np.argwhere((X_nodes[:] == node) & (X[:,Tree_Nodes[node][1]]>=Tree_Nodes[node][2]))] = Tree_Nodes[node][4]
        # If the node is a terminal node, the points are used to label regions
        if Tree_Nodes[node][0] == True:
            try:
                y_val_pre[np.argwhere(X_nodes[:]==node)] = R_values[Tree_Nodes[node][1]]
            except:
                print()
    return y_val_pre   
    
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
                   departure_time,stops,arrival_time,
                   flclass,duration,days_left,tree_regions)).astype(float)

y_all = price.astype(float)

# Standardizing the predictors and the response
# X_all[:,0:-1] = standardize(X_all[:,0:-1])
# y_all = standardize(y_all)
   



# Performing a test-train split
X_tr, X_te, y_tr, y_te = train_test_split(X_all,y_all,0.2,42)

max_reg = 1000
min_points = 100

RMSE_cv = 0

# Creating the tree structure based on minimizing RSS, the algorithm will 
# attach an integer region data to all of the data points in the training data
Tree_Nodes, X_train = Decision_Tree_Constructer(X_tr,y_tr,max_reg,min_points)

# Finding the mean for each region by using the price values associated with 
# the points in that region
R_values = find_R_values(X_tr,y_tr)

# Assigning region values to the points in the validation data set based on 
# the nodes designated by the tree algorithm and using the mean values inside 
# the regions to predict the points in the validation set
y_te_pre = predict(X_te,Tree_Nodes,R_values)

# Computing the MSE for the predictions and real price values on the valida-
# tion data set
MSE_te = find_RSS(np.squeeze(y_te),np.squeeze(y_te_pre))/y_te.shape[0]
RMSE = np.sqrt(MSE_te)
    
print("RMSE for the test data \
      (using {mr} max regions): \n".format(mr = max_reg+1))
print(RMSE)


#%%




