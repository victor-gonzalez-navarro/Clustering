import numpy as np


class Kmeans:
    labels_km = None

    def kmeans_method(self, data_x):
        print('\n'+'Computing clusters with K-means Clustering')
        result_sse = []
        result_labels = []
        for nm in range(0, self.num_tries_init):
            centroids = data_x[np.random.randint(0, len(data_x) - 1, self.num_clusters)]
            result_sse,result_labels = self.kmeans_algorithm(data_x, self.num_clusters, self.max_iterations, centroids,
                                                result_sse, result_labels)

        print('Accuracy with initalization: '+str(np.argmin(result_sse))+' (the best one)')
        self.labels_km = result_labels[np.argmin(result_sse)]

    def kmeans_algorithm(self, data_x, n_clusters, max_iterations, centroids, result_sse, result_labels):
        n_instances = data_x.shape[0]
        n_features = data_x.shape[1]
        resta = np.zeros((n_instances, n_clusters))

        for iterations in range(0, max_iterations):
            new_centroids = np.zeros((n_clusters, n_features))
            m_instpercluster = [0] * n_clusters

            for i in range(0, n_clusters):
                resta[:, i] = np.sum((data_x[:, 0:n_features] - centroids[i, :]) ** 2, axis=1)

            SSE = np.sum(np.min(resta, axis=1))
            print('SSE in iteration ' + str(iterations) + ' is: ' + str(SSE))
            lista = np.argmin(resta, axis=1)

            for i in range(0, n_clusters):
                info = data_x[np.argwhere(lista == i).reshape(np.argwhere(lista == i).shape[0], ), 0:n_features]
                new_centroids[i, :] = np.sum(info, axis=0)
                m_instpercluster[i] = np.sum(lista == i)
                centroids[i, :] = new_centroids[i, :] / m_instpercluster[i]

        print('SSE for specific initialization ' + ' --> ' + str(SSE)+'\n')
        result_sse.append(SSE)
        result_labels.append(lista)
        return result_sse,result_labels

    def __init__(self, num_clusters, num_tries_init, max_iterations):
        self.num_clusters = num_clusters
        self.num_tries_init = num_tries_init
        self.max_iterations = max_iterations