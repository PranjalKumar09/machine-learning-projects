# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/machine-learning-projects
# ===============================

""" 
wcss -> within clusters sum of squares 


"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 

customer_data  = pd.read_csv('Datasets/csv/Mall_Customers.csv')
# print(customer_data.shape) # (200, 5)
# there is no null values 

X = customer_data.iloc[:,[3,4]].values
# print(X)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init='k-means++' , random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
sns.set()
plt.plot(range(1,11) , wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()




# optimum number of clusters = 5
# training the k-means clustering model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
# return a label for each data point based on their cluster 
Y = kmeans.fit_predict(X)



plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0] , X[Y==0,1] , s = 50 , c='green' , label = 'Cluster 1') 
plt.scatter(X[Y==1,0], X[Y==1,1], s = 50, c='red' , label = 'Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s = 50, c='yellow' , label = 'Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s = 50, c='violet' , label = 'Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s = 50, c='blue' , label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[: , 0] , kmeans.cluster_centers_[: , 1] , s = 100 , c = 'cyan', label = 'Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
