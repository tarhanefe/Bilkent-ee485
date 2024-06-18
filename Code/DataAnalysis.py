# Importing all the Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Displaying the head of the dataset 
df = pd.read_csv("/Users/efetarhan/Desktop/DATA/Clean_Dataset.csv")
df = df.drop('Unnamed: 0', axis=1)


#%% Displaying the correlation of the dataset 

def preprocs(df):
    
    
    #df["stops"] = df["stops"].replace({'zero':0,'one':1,'two_or_more':2}).astype(int)
    #df["class"] = df["class"].replace({'Economy':0,'Business':1}).astype(int)
    
    stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
    class_mapping = {'Economy': 0, 'Business': 1}

    df['stops'] = df['stops'].map(stops_mapping).astype(int)
    df['class'] = df['class'].map(class_mapping).astype(int)
    
    dummies_variables = ["airline","source_city","destination_city","departure_time","arrival_time"]
    dummies = pd.get_dummies(df[dummies_variables], drop_first= True)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop(["flight","airline","source_city","destination_city","departure_time","arrival_time"],axis=1)
    return df

df_preprocessed = preprocs(df)
plt.figure(dpi = 600,figsize=(20,10))
sns.heatmap(df_preprocessed.corr(),annot=True,cmap='Greens')
plt.title("Correlation Map of the Features in the Dataset")
plt.show()


plt.figure(dpi = 600,figsize=(15,5))
sns.heatmap(df.corr(),annot=True,cmap='Greens')
plt.title("Correlation Map of the Numerical Features in the Dataset")
plt.show()


#%% Price range of 6 airlines

plt.figure(figsize=(15,5),dpi = 600)
sns.boxplot(x=df['airline'],y=df['price'],palette='hls')
plt.title('Price Range of 6 Airline Compaines',fontsize=13)
plt.xlabel('Airline',fontsize=16)
plt.ylabel('Price ($)',fontsize=16)
plt.show()


#%% Days Left vs Ticket Price
plt.figure(dpi = 600, figsize=(20,8))
sns.lineplot(data=df,x='days_left',y='price',color='red')
plt.title('Days Left For Departure vs Ticket Price Plot')
plt.xlabel('Days Left for Departure')
plt.ylabel('Price ($)')
plt.show()


#%% Pie chart of portions 

df2=df.groupby(['flight','airline','class'],as_index=False).count()
plt.figure(figsize=(10,8),dpi = 600)
df2['class'].value_counts().plot(kind='pie',autopct='%.2f',cmap='coolwarm',labels=['']*len(df2['class'].value_counts()))
plt.title('Flight Class Portions in the Dataset')
plt.legend(['Economy','Business'])
plt.show()


#%% Class vs Price Plot

plt.figure(dpi = 600, figsize=(10,5))
sns.violinplot(x='class',y='price',data=df)
plt.title('Class Vs Ticket Price',fontsize=15)
plt.xlabel('Class',fontsize=15)
plt.ylabel('Price($)',fontsize=15)
plt.xticks(ticks=[0, 1], labels=['Economy', 'Business'])
plt.show()

#%% Departure and Arrival Time vs Price Plot

plt.figure(figsize=(24,10))
plt.subplot(1,2,1)
sns.boxplot(x='departure_time',y='price',data=df,palette = 'coolwarm')
plt.title('Departure Time Vs Ticket Price',fontsize=20)
plt.xlabel('Departure Time',fontsize=15)
plt.ylabel('Price($)',fontsize=15)
plt.subplot(1,2,2)
sns.boxplot(x='arrival_time',y='price',data=df,palette='Blues')
plt.title('Arrival Time Vs Ticket Price',fontsize=20)
plt.xlabel('Arrival Time',fontsize=15)
plt.ylabel('Price($)',fontsize=15)
plt.show()


#%% Source City and Destination City vs Price

plt.figure(dpi = 600, figsize=(24,10))
plt.subplot(1,2,1)
sns.boxplot(x='source_city',y='price',data=df)
plt.title('Source City vs Ticket Price',fontsize=20)
plt.xlabel('Source City',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.subplot(1,2,2)
sns.boxplot(x='destination_city',y='price',data=df,palette='coolwarm_r')
plt.title('Destination City vs Ticket Price',fontsize=20)
plt.xlabel('Destination City',fontsize=15)
plt.ylabel('Price',fontsize=15)


#%% Days Left for Departure vs the Price Plot 
plt.figure(dpi = 600,figsize=(20,8))
sns.lineplot(data=df,x='days_left',y='price',color='blue',hue='airline',palette='hls')
plt.title('Days Left For Departure Versus Ticket Price of Airlines',fontsize=15)
plt.legend()
plt.xlabel('Days Left',fontsize=15)
plt.ylabel('Price($)',fontsize=15)
plt.show()


