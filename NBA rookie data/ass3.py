#import necessary libraries
import pandas as pd #For processing
#Import for visualization
import seaborn as sns 
import matplotlib.pyplot as plt
#Machine learning modules
from sklearn.model_selection import train_test_split #for splitting the data
from sklearn.metrics import accuracy_score # To check performance
from sklearn.model_selection import RandomizedSearchCV #for imporovements
#Selected Algorithms employed
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

#load the dataset
df=pd.read_csv("nba_rookie_data_-2117739787.csv")

print(df.shape) # Check the shape of the data
print(df.head()) # Display first five rows
print(df.tail()) # Display the last five 
print(df.columns) # Check the columns
print(df.isna().sum()) # Check for null values
print(df.info()) # Check summarized info of the data
print(df.describe()) # Check the statistical information of the 
print(df.corr()) # Check the correlation
print(df.duplicated().sum()) # Check for duplicates

df.dropna(inplace=True) #Drop the null values

df.drop_duplicates(keep='first',inplace=True) #Remove the duplicates

# Manchine Learning

#Get feature and target
X = df.iloc[:,1:-1]
y=df.iloc[:,-1]

#Display feature and target
print(X)
print(y)
features = list(X.columns)
print(features)

models = [LogisticRegression(), GaussianNB(),MLPClassifier()] # list holding instances of model
# List to hold each model's metrics
accuracy_scores1 = []
mis_clas1 = []

X=df[features] #Get X
#Splitting the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#Iterate through the models
for model in models:

    #Fit the model
    model.fit(X_train,y_train)

    #make prediction on test data
    y_pred = model.predict(X_test)

    # Get the metrics
    score=round(model.score(X_test,y_test),3)
    mis_cla=(X_test.shape[0],(y_test!=model.predict(X_test)).sum())[1]
    
    #Store the metric in the list
    accuracy_scores1.append(score)
    mis_clas1.append(mis_cla)
    
    #Display metrics i.e effectiveness
    print(f"{model} Effectiveness")
    print(score)
    print(mis_cla)
    print()


#Trying selected features to observe the changes
features = ["Games Played","Minutes Played","Points Per Game","Rebounds","Assists"]
X=df[features]
y=df.iloc[:,-1]

#Splitting the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#List to hold metrics
accuracy_scores2 = []
mis_clas2 = []


#Iterate through the models
for model in models:

    #Fit the model
    model.fit(X_train,y_train)

    #make prediction on test data
    y_pred = model.predict(X_test)
    
    # Get the metrics
    score = round(model.score(X_test,y_test),3)
    mis_cla = (X_test.shape[0],(y_test!=model.predict(X_test)).sum())[1]
    
    #Store the metric in the list
    accuracy_scores2.append(score)
    mis_clas2.append(mis_cla)
    
    #Display metrics i.e effectiveness
    print(f"{model} Effectiveness")
    print(score)
    print(mis_cla)
    print()


# Store the metrics in a dictionary
models = {
    "acc_score":accuracy_scores1 + accuracy_scores2,
    "models":["log",'gnb','mlp',"log",'gnb','mlp'],
    "acc":["acc1","acc1","acc1","acc2","acc2","acc2"],
    "mis_cla": mis_clas1 + mis_clas2,
    "mcla": ["mcla1","mcla1","mcla1","mcla2","mcla2","mcla2"]
}

#load the dictionary to a pandas dataframe
mydata = pd.DataFrame(models)
print(mydata)

# Plot the accuracy scores
sns.set()
plt.figure(figsize=(10,6))
sns.barplot("models" , "acc_score",hue="acc",data=mydata)
plt.title("Accuracy Scores")
plt.savefig("acc_score.png")
plt.show()

# Plot the misclassified points
sns.set()
plt.figure(figsize=(10,6))
sns.barplot("models" , "mis_cla",hue="mcla",data=mydata)
plt.title("Misclassified points")
plt.savefig("misclas.png")
plt.show()

#Improvements on the model

X = df.iloc[:,1:-1]
features = list(X.columns)
X=df[features]
y=df.iloc[:,-1]

#Splitting the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#Hypertuning logistic regression
logreg = LogisticRegression()
parameters = {
    "C":[0.001,0.1,1.0,0.01,100,10.0],
    "penalty":["none","l2","l1","elasticnet"],
    "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
rand = RandomizedSearchCV(estimator=logreg, param_distributions=parameters,cv=5) # Applying Randomized Cross Validation

rand.fit(X,y)
print("The best parameters: ",rand.best_params_)

#apply the best params
logreg = LogisticRegression(penalty = 'l1',C = 0.1,solver = 'liblinear',max_iter=1000)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#Display the metrics
logacc = accuracy_score(y_test,y_pred)
print("The acccuracy score is: ",round(logacc,3))
log_mis_cla = (X_test.shape[0],(y_test!=y_pred).sum())[1]
print("The misclassified points: ",log_mis_cla)


# Hypertuning the GaussianNB
gnb = GaussianNB()
parameters = {
    "priors":[[0.3, 0.7], [0.5, 0.5], [0.2, 0.4, 0.4]],
    "var_smoothing":[1e-7,1e-6,1e-5]
}

rand = RandomizedSearchCV(estimator=gnb, param_distributions=parameters,cv=5) # Applying Randomized Cross Validation

rand.fit(X,y)
print("The best parameters: ",rand.best_params_)

#apply the best params
gnb = GaussianNB(var_smoothing = 1e-07,priors = [0.3, 0.7])
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

#Display the metrics
gnbacc = accuracy_score(y_test,y_pred)
print("The acccuracy score is: ",round(gnbacc,3))
gnb_mis_cla = (X_test.shape[0],(y_test!=y_pred).sum())[1]
print("The misclassified points: ",gnb_mis_cla)

# Hypertuning the MLP classifier
mlp = MLPClassifier()
parameters = {
    "hidden_layer_sizes":[(10, 50, 20), (20,10,40), (20,50,100)],
    "activation":["relu","logistic"]
}
rand = RandomizedSearchCV(estimator=mlp, param_distributions=parameters,cv=5)

rand.fit(X,y)
print("The best parameters: ",rand.best_params_)

# Apply the best params
mlp = MLPClassifier(hidden_layer_sizes=(20,50,100),activation = "logistic",
                   random_state = 4, max_iter = 2000)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)

#Display the metrics
mlpacc = accuracy_score(y_test,y_pred)
print("The acccuracy score is: ",round(mlpacc,3))
mlp_mis_cla = (X_test.shape[0],(y_test!=y_pred).sum())[1]
print("The misclassified points: ",mlp_mis_cla)

#Creat a list hold their accuracy score
acc_score3 = []
acc_score3.append(logacc)
acc_score3.append(gnbacc)
acc_score3.append(mlpacc)

#List to hold thier misclassification
mis_clas3 = []
mis_clas3.append(log_mis_cla)
mis_clas3.append(gnb_mis_cla)
mis_clas3.append(mlp_mis_cla)

# Store the metrics in a dictionary
models = {
    "acc_score":accuracy_scores1 + acc_score3,
    "models":["log",'gnb','mlp',"log",'gnb','mlp'],
    "acc":["acc1","acc1","acc1","tuned","tuned","tuned"],
    "mis_cla": mis_clas1 + mis_clas3,
    "mcla": ["mcla1","mcla1","mcla1","tuned","tuned","tuned"]
}

#load the dictionary to a pandas dataframe
mydata = pd.DataFrame(models)
print(mydata)

#Plot the accuracy after tuning
sns.set()
plt.figure(figsize=(10,6))
sns.barplot("models" , "acc_score",hue="acc",data=mydata)
plt.title("Accuracy Scores after Hypertuning")
plt.savefig("acc_score_hypertuned.png")
plt.show()

#Plot the misclassified points fater hypertuning
sns.set()
plt.figure(figsize=(10,6))
sns.barplot("models" , "mis_cla",hue="mcla",data=mydata)
plt.title("Misclassified points after Hypertuning")
plt.savefig("misclas_hypertuned.png")
plt.show()

#Correlation matrix showing predictor and target relationship
plt.figure(figsize=(20,7))
sns.heatmap(df.corr(),annot = True)
plt.title("Relationship between Target and Predictor")
plt.savefig("relationship.png")
plt.show()

#Predict a single value with the best instances of our model

logreg = LogisticRegression()
gnb = GaussianNB(var_smoothing = 1e-07,priors = [0.3, 0.7])
mlp = MLPClassifier(hidden_layer_sizes=(20,50,100),activation = "logistic",
                   random_state = 4, max_iter = 2000)

#fit values
logreg.fit(X_train,y_train)
gnb.fit(X_train,y_train)
mlp.fit(X_train,y_train)

#Data to predict
input_data = [36,27.4,7.4,2.6,7.6,34.7,0.5,2.1,25.0,1.6,2.3,69.9,0.7,3.4,4.1,1.9,0.4,0.4,1.3]

print("A single value is: ",logreg.predict([input_data])[0])
print("A single value is: ",gnb.predict([input_data])[0])
print("A single value is: ",mlp.predict([input_data])[0])