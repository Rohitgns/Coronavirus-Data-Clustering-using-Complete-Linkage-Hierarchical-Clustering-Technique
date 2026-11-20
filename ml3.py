"""
Roll no - 20CS60R71
Name - Rohit
Project no - DC3
project title - Coronavirus Data Clustering using Complete Linkage Hierarchical Clustering Technique
"""


import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means

# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]

dataset = pd.read_csv('COVID_3_unlabelled.csv',encoding='latin-1')
dataset = dataset.iloc[:, 1:].values
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
# standardize dataset
standardize_dataset(dataset, means, stdevs)
 
def euclidean_distance(a, b):
    return [[sqrt(sum((e1-e2)**2 for e1,e2 in zip(ee1,ee2))) for ee2 in b] for ee1 in a]
    
#Defining our function
def kmeans(x,k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    distances = euclidean_distance(x, centroids)
     
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = euclidean_distance(x, centroids)
        points = np.array([np.argmin(i) for i in distances])
         
    return points 
k = 6
df = dataset
label = kmeans(df,k,20)
#Visualize the results

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()
k3file = open("k3points.txt", "w")
k3file.write(str(label))
k3file.close()

def silhouetter_score(dataset, label, k):
    silhouette_data = []
    no_of_instances = len(dataset)

    for row in range(0, no_of_instances):

        this_instance = dataset[row]
        this_cluster = label[row]

        a = None
        running_sum = 0.0
        counter = 0.0

        for row_2 in range(0, no_of_instances):

            if this_cluster == label[row_2]:

                distance = np.linalg.norm(this_instance - dataset[row_2])

                running_sum += distance
                counter += 1

        if counter > 0:
            a = running_sum / counter

        b = float("inf")

        for clstr in range(0, k):

            running_sum = 0.0
            counter = 0.0

            if clstr != this_cluster:
                for row_3 in range(0, no_of_instances):

                    if label[row_3] == clstr:

                        distance = np.linalg.norm(this_instance - dataset[row_3])
                        running_sum += distance
                        counter += 1

                if counter > 0:
                    avg_distance_to_cluster = running_sum / counter

                if avg_distance_to_cluster < b:
                    b = avg_distance_to_cluster

        s = (b - a) / max(a,b)
        silhouette_data.append(s)
    return sum(silhouette_data) / len(silhouette_data)

funscore = silhouetter_score(dataset, label, k)
print(f"silhouette score = {funscore}")
import random

k = 5

class cluster_node:
    def __init__(self, vec, id, left=None, right=None, distance=0.0, node_vector = None):
        self.leftnode = left
        self.rightnode = right
        self.vec = vec
        self.id = id
        self.distance = distance
        if node_vector is None:
            self.node_vector = [self.id]
        else:
            self.node_vector = node_vector[:]

def euclidean_distance(vec1, vec2):
    return np.sqrt(sum((vec1 - vec2) ** 2))
def max_distance(clust1, clust2, distances):
    d = 0
    for i in clust1.node_vector:
        for j in clust2.node_vector:
            try:
                distance = distances[(i,j)]
            except:
                try:
                    distance = distances[(i,j)]
                except:
                    distance = euclidean_distance(clust1.vec, clust2.vec)
            if distance > d:
                d = distance
    return d


def agglomerative_clustering(data, distance):
    # cluster the rows of the data matrix
    distances = {}
    currentclustid = -1

    # cluster nodes are initially just the individual rows
    nodes = [cluster_node(np.array(data[i]), id=i) for i in range(len(data))]

    while len(nodes) > k:
        lowestpair = (0,1)
        closest = euclidean_distance(nodes[0].vec,nodes[1].vec)
    
        # loop through every pair looking for the smallest distance
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                # distances is the cache of distance calculations
                if (nodes[i].id,nodes[j].id) not in distances:
                    distances[(nodes[i].id,nodes[j].id)] = max_distance(nodes[i], nodes[j], distances)
        
                d = distances[(nodes[i].id,nodes[j].id)]
        
                if d < closest:
                    closest = d
                    lowestpair = (i,j)
        
        # calculate the average of the two nodes
        len0 = len(nodes[lowestpair[0]].node_vector)
        len1 = len(nodes[lowestpair[1]].node_vector)
        mean_vector = [(len0*nodes[lowestpair[0]].vec[i] + len1*nodes[lowestpair[1]].vec[i])/(len0 + len1) \
                        for i in range(len(nodes[0].vec))]
        
        # create the new cluster node
        new_node = cluster_node(np.array(mean_vector), currentclustid, left = nodes[lowestpair[0]], right = nodes[lowestpair[1]], \
            distance = closest, node_vector = nodes[lowestpair[0]].node_vector + nodes[lowestpair[1]].node_vector)
        
        # cluster ids that weren't in the original set are negative
        currentclustid -= 1
        del nodes[lowestpair[1]]
        del nodes[lowestpair[0]]
        nodes.append(new_node)

    return nodes

def main():
    # Generate data
    colorset = ['red', 'green', 'blue', 'yellow', 'brown', 'orange', 'black']
    # df = pd.read_csv('./segmentation.data.modified')
    idx = np.random.choice(len(dataset), k, replace=False)
    #Randomly choosing Centroids 
    centers = dataset[idx, :]
    df = dataset

    data = np.array(df)

    # Average criterion agglomerative clustering
    cluster = agglomerative_clustering(data, "max")
    j = 0
    m = plt.figure(2)
    for i in cluster:
        plt.scatter(data[i.node_vector].T[0], data[i.node_vector].T[1], color = colorset[j])
        j += 1
    m.show()

if __name__ == '__main__':
    main()
