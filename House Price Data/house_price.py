#import the necessary libraries
import numpy as np #numerical computation
import pandas as pd #for processing
import matplotlib.pyplot as plt #Visuals
import seaborn as sns #Visuals
from sklearn.preprocessing import StandardScaler #For scaling
from sklearn.model_selection import train_test_split # import train test split function
from sklearn.linear_model import LinearRegression # import the linear regression
from sklearn.metrics import  r2_score, mean_squared_error #metrics


#load the dataset
df=pd.read_csv("houseprice_data.csv")

print(df.head()) # Display first five rows
print(df.tail()) # Display the last five 
print(df.columns) # Check the columns
print(df.isna().sum()) # Check for null values
print(df.info()) # Check summarized info of the data
print(df.describe()) # Check the statistical information of the 
print(df.corr()) # Check the correlation
print(df.shape) # Check the shape of the data
print(df.duplicated().sum()) # Check for duplicates


df.drop_duplicates(keep='first',inplace=True) # Remove duplicates

# check if there are rows where the square footage of living space in the neighborhood is greater than the square footage of the lot in the neighborhood.
df[df['sqft_living15']>df['sqft_lot15']]

# check if there are rows where the square footage of living space in the house is greater than the square footage of the lot on which the house is situated.
df[df['sqft_living']>df['sqft_lot']]

# select rows where the square footage of living space in the house is greater than the square footage of the lot on which the house is situated.
df=df[df['sqft_living']<df['sqft_lot']]

# select rows where the square footage of living space in the neighborhood is greater than the square footage of the lot in the neighborhood.
df=df[df['sqft_living15']<df['sqft_lot15']]

X=df.iloc[:,1:] # Seperate the features from target

#Select the columns name as feature and convert it from array to list
features=X.columns
features=list(features)

#Create list for mse and r2_score
mse_scores=[]
r2_scores=[]
models=[]


#Create a function to go through diffferent number of features
def tweaking_features():

    num_feat=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','view',
        'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

    cat_feat=['waterfront','zipcode']
    
     # Copy the features list to avoid modifying the original list
    updated_features = features.copy()

    for i in range (len(features)):
        
        #Display the feature
        print("The current numerical features are ",num_feat)
        print("The current categorical features are ",cat_feat)

        scaler=StandardScaler() # Create an instance of the scaler

        X=df[updated_features] #Select the features
        y=df.iloc[:,0] #select the target variable

        #Splitting the dataset into train and test data
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

        X_train[num_feat]=scaler.fit_transform(X_train[num_feat])
        X_test[num_feat]=scaler.transform(X_test[num_feat])

        model=LinearRegression() # Create an instance

        model.fit(X_train,y_train) # Fit the model

        y_pred = model.predict(X_test) # make prediction on test data

        r_score=round(r2_score(y_test,y_pred),4) #Check the r2 score
        mse=round(mean_squared_error(y_test,y_pred),3) # Check the mse
        
        #Store the metrics and the model in their respective list 
        r2_scores.append(r_score)
        mse_scores.append(mse)
        models.append(f"m{i+1}")
        
        #Display the metrics
        print(f"Instance {i+1}")
        print("The r2_score is: ",r_score)
        print("The mean_squared_error is: ",mse)
        
         # Select the feature to be removed in this iteration
        removed = updated_features.pop(-1)
        
        #check if the removed feature is in the numerical feature list
        if removed in num_feat:
            num_feat.remove(removed) #remove it from the num feature list
        else:
            cat_feat.remove(removed) #remove it from the categorical feature list
        
        print() #Display a space to seperate each iteration


tweaking_features() # Call the function


# Effectiveness of each model

#Create a dictionary to hold the metrics
metrics={"r2_scores": r2_scores,
        "mse": mse_scores,
         "models":  models
    }

#Load it into a dataframe
data=pd.DataFrame(metrics)

#sort the values
top5=data.sort_values(by=['r2_scores'],ascending=False).head(5)

#Visualize and save the Rsquared score of top5 models
plt.figure(facecolor='white',figsize=(8,6))
sns.barplot(y=top5['r2_scores'],x=top5['models'])
plt.title("R_squared score of Top 5 models based on their features")
plt.savefig("top 5.png")

#Visualize and save the mse of the top5 models
plt.figure(facecolor='white',figsize=(8,6))
sns.barplot(y=top5['mse'],x=top5['models'])
plt.title("MSE of Top 5 models based on their features")
plt.savefig("top 5 MSE.png")

#Removing the "floor" and "sqft_basement" feature
features=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

#num features
num_feat=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'view',
        'condition', 'grade', 'sqft_above',
        'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

scaler=StandardScaler() #instance of scale

X=df[features]
y=df.iloc[:,0]

#Splitting the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

# Scale the features
X_train[num_feat]=scaler.fit_transform(X_train[num_feat])
X_test[num_feat]=scaler.transform(X_test[num_feat])

reg=LinearRegression() # Create an instance

reg.fit(X_train,y_train) # Fit the model


y_pred = reg.predict(X_test) # Make prediction on test data

# Get metrics
reg_r2=round(r2_score(y_test,y_pred),4)
reg_mse=round(mean_squared_error(y_test,y_pred),3)

# Display metrics
print("The r2_score is: ",reg_r2)
print("The mean_squared_error is: ",reg_mse)

#Compare the metrics of m1 and reg
data2={
    'model': ["m1","reg"],
    "mse": [mse_scores[0],reg_mse],
    "r2_score" :[r2_scores[0],reg_r2]
}


compare=pd.DataFrame(data2) # load the data to dataframe

#Visualize the metrics 
plt.figure(facecolor='white',figsize=(8,6))
sns.barplot(y=compare['mse'],x=compare['model']) #mse visualization
plt.title("mse score of m1 and reg")
plt.savefig("compare mse.png")

plt.figure(facecolor='white',figsize=(8,6))
sns.barplot(y=compare['r2_score'],x=compare['model']) #r2_score visualization
plt.title("R_squared score of m1 and reg")
plt.savefig("compare r2_score.png")

#Prediction a single value
prediction = reg.predict(np.array([[4,1,1208,4590,0,2,4,5,1208,2013,0,98178,47.5114,-112.253,1203,5650]]))
print('Your house price costs: ',format(prediction[0],',.2f'))
