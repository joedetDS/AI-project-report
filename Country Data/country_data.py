#Import the necessary librararies
import pandas as pd #for processing and cleaning
#import for visualization
import matplotlib.pyplot as plt
import seaborn as sns
#machine learning module for clustering
from sklearn.cluster import KMeans


df = pd.read_csv("country_data_-75368203.csv") #load the dataset

print(df.head()) # Display first five rows
print(df.tail()) # Display the last five 
print(df.columns) # Check the columns
print(df.isna().sum()) # Check for null values
print(df.info()) # Check summarized info of the data
print(df.describe()) # Check the statistical information of the 
print(df.corr()) # Check the correlation
print(df.shape) # Check the shape of the data
print(df.duplicated().sum()) # Check for duplicates

#Choosing the Child Mortality and Life Expectancy
X = df.iloc[:,[1,7]].values
print(X) # Display the selected features

wcss1=[] #list to hold all wcss

#Function to get the optimum wcss
def get_optimum(wcss):
    #Choosing the best number of clusters using WCSS- Within Clusters Sum of Squares

    #loop through 10 possible clusters
    for i in range(1,11):
        # create an instance
        kmeans=KMeans(n_clusters = i, init='k-means++',random_state=3)
        kmeans.fit(X)

        #append the wcss values
        wcss.append(kmeans.inertia_)

#Call the function by passing the list to it
get_optimum(wcss1)
print("The wcss values are : \n",wcss1) # Display the wcss values


#Plotting an elbow graph to view the wcss
sns.set()
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss1)
plt.title("Elbow Graph of WCSS1")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig('wcss1.png')
plt.show()

# From the graph the significant drop in value occured in 2 and 3
#Hence the optimum number of clusters is 3
#Training the k_means model by specifying 3 as the number of clusters
kmeans = KMeans(n_clusters = 3,init='k-means++',random_state=2)
# returning the labels of each clusters formed
y = kmeans.fit_predict(X)
print(y) # Display the labels

#Visualizing the clusters
plt.figure(figsize=(10,6))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c="green",label="Cluster 1")
plt.scatter(X[y==1,0],X[y==1,1],s=50,c="yellow",label="Cluster 2")
plt.scatter(X[y==2,0],X[y==2,1],s=50,c="red",label="Cluster 3")
#Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100,c='cyan',label='Centroids')
#Labels
plt.xlabel("Child Mortality")
plt.ylabel("Life Expectancy")
plt.title("Mortality Groups")
plt.savefig('Cluster1.png')
plt.show()


#Income and the gdpp
X=df.iloc[:,[5,9]].values
print(X)

wcss2=[] #list to hold all wcss

#Call the function by passing the list to it
get_optimum(wcss2)

print("The wcss values are : \n",wcss2)# Display the wcss values

#Plotting an elbow graph to view the wcss
sns.set()
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss2)
plt.title("Elbow Graph of WCSS2")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig('wcss2.png')
plt.show()

#Significant drops occurs at 2,3 and 4
#The optimal wcss is 4
kmeans = KMeans(n_clusters = 4,init = 'k-means++',random_state=3)
#return the label of each clusters
y = kmeans.fit_predict(X)
print(y) # Display the labels
centroids = kmeans.cluster_centers_
print("The centroids are :",centroids)

#Visualizing the clusters
plt.figure(figsize=(10,6))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c="green",label="Cluster 1")
plt.scatter(X[y==1,0],X[y==1,1],s=50,c="red",label="Cluster 2")
plt.scatter(X[y==2,0],X[y==2,1],s=50,c="yellow",label="Cluster 3")
plt.scatter(X[y==3,0],X[y==3,1],s=50,c="violet",label="Cluster 3")
#Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100,c='cyan',label='Centroids')
#Labels
plt.xlabel("Income")
plt.ylabel("GDPP")
plt.title("Income by GDPP Grouping")
plt.savefig("Cluster2.png")
plt.show()


# Income, GDPP and health
X = df.iloc[:,[3,5,9]].values
print(X)

wcss3=[] #list to hold all wcss

#Call the function by passing the list to it
get_optimum(wcss3)

print("The wcss values are : \n",wcss3) # Display the wcss values

#Plotting an elbow graph to view the wcss
sns.set()
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss2)
plt.title("Elbow Graph of WCSS3")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("wcss3.png")
plt.show()

#Significant drops occurs at 2,3 and 4
#The optimal wcss is 4
kmeans = KMeans(n_clusters = 4,init = 'k-means++',random_state=3)
#return the label of each clusters
y = kmeans.fit_predict(X)
print(y) # Display the labels
centroids = kmeans.cluster_centers_ #Get the centroids
print("The centroids are :",centroids)

# Create a 3D figure
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
# Create the 3D scatter plot
ax.scatter(X[y==0,0], X[y==0,1], X[y==0,2], c='green')
ax.scatter(X[y==1,0], X[y==1,1], X[y==1,2], c='red')
ax.scatter(X[y==2,0], X[y==2,1], X[y==2,2], c='blue')
ax.scatter(X[y==3,0], X[y==3,1], X[y==3,2], c='yellow')
# Set labels for the axes
ax.set_xlabel('Health')
ax.set_ylabel('Income')
ax.set_zlabel('GDPP')
#Plot the centroids
ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2], s=100,c='cyan',label='Centroids')
plt.title("Economic Wellbeing and Health Status Grouping")
plt.savefig("Cluster3.png")
# Show the plot
plt.show()
