import numpy as np
import pandas as pd
import random

from numpy.linalg import norm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, K):
        self.K = K

    def initializ_centroids(self, X):
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.K]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.K):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(100):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)

    def predict(self, X,centroids):
        distance = self.compute_distance(X, centroids)
        return self.find_closest_cluster(distance)

def load_data(file_name):
    data = pd.read_csv(file_name)
    data = data.drop(["Unnamed: 0"], axis=1)
    return data


def split_data(data, rate=0.2):
    data_train, data_test = train_test_split(data, test_size=rate)
    return data_train, data_test


def train_test_separate(data):
    X = data.drop('Private', axis=1)
    y = data['Private']
    return X

# Elbow method
def get_best_K(data):
    error_list = []
    K = np.arange(1, 10)
    for k in K:
        kmean_model = Kmeans(k)
        kmean_model.fit(data)
        error = kmean_model.error
        error_list.append(error)
    # Plot the elbow
    plt.plot(K, error_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


if __name__ == '__main__':
    data = load_data("College.csv")
    data_train, data_test = split_data(data)

    X_train = train_test_separate(data_train)
    X_test  = train_test_separate(data_test)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)

    ##  Choose best K from graph
    get_best_K(X_train)
    best_K = 5

    print("From implementation :")
    kmean_model = Kmeans(best_K)
    kmean_model.fit(X_train)
    centroids = kmean_model.centroids
    labels_pred = kmean_model.predict(X_test, centroids)
    error_1 = kmean_model.compute_sse(X_test,labels_pred, centroids)
    print(centroids)
    print("Error 1 : {error}".format(error = error_1))



    print("From library :")
    kmeans = KMeans(n_clusters=best_K).fit(X_train)
    centroids_library = kmeans.cluster_centers_
    print(centroids_library)
    predict_labels = kmeans.predict(X_test)
    error_2 = kmean_model.compute_sse(X_test,predict_labels,centroids_library)
    print("Error 2 : {error}".format(error=error_2))



