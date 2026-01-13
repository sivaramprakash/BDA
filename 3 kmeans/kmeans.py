import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

 

dataset = pd.read_csv('../dataset/country.csv')

 

print(dataset.shape)

print(dataset.describe())

print(dataset.head(5))

 

GDP = dataset['GDP'].values

Literacy = dataset['LITERACY'].values

X = np.array(list(zip(GDP, Literacy)))

 

from sklearn.cluster import KMeans

 

wcv = []

for i in range(1, 11):

    km = KMeans(n_clusters=i, random_state=0)

    km.fit(X)

    wcv.append(km.inertia_)

 

plt.plot(range(1, 11), wcv, color="red", marker="8")

plt.title("Optimal K value (Elbow Method)")

plt.xlabel("No. of Clusters")

plt.ylabel("WCV")

plt.show()

 

model = KMeans(n_clusters=3, random_state=0)

y_means = model.fit_predict(X)

 

plt.scatter(GDP, Literacy, c=model.labels_)

plt.show()

 

plt.scatter(X[y_means==0,0], X[y_means==0,1], s=50, c='purple', label='Cluster 1')

plt.scatter(X[y_means==1,0], X[y_means==1,1], s=50, c='green', label='Cluster 2')

plt.scatter(X[y_means==2,0], X[y_means==2,1], s=50, c='blue', label='Cluster 3')

 

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, marker='s', c='red', label='Centroids')

plt.title('Country Clustering Based on GDP & Literacy')

plt.xlabel('GDP')

plt.ylabel('Literacy')

plt.legend()

plt.show()