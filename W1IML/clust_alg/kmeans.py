import numpy as np
from matplotlib import pyplot
from eval_plot.evaluation import ploting_v

class Kmeans:
    labels_km = None

    def _has_converged(self, prev_sse, curr_sse):
        return abs(curr_sse - prev_sse) <= 0.1

    # Main algorithm
    def kmeans_method(self, data_x):
        print('\n'+'\033[1m'+'Computing clusters with K-means algorithm...'+'\033[0m')

        # Local variables
        result_sse = []
        result_labels = []

        # Compute kmeans for different initialization of the clusters
        for nm in range(0, self.num_tries_init):
            # Random initialization of the centroids
            centroids = data_x[np.random.randint(0, len(data_x) - 1, self.num_clusters)]
            result_sse,result_labels = self.kmeans_algorithm(data_x, self.num_clusters, self.max_iterations, centroids,
                                                result_sse, result_labels)

        # Show the accuracy obtained with the best initialization
        print('\033[1m'+'Accuracy with initalization: '+str(np.argmin(result_sse))+' (the best one)'+'\033[0m')
        self.labels_km = result_labels[np.argmin(result_sse)]
        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(min(result_sse),
                                                                                          2)) + '\033[0m')

        # Scatter plot
        # ploting_v(data_x, self.num_clusters, self.labels_km)
        pyplot.figure()
        colour = iter(pyplot.cm.rainbow(np.linspace(0, 1, 4)))
        for i in range(4):
            pyplot.plot(data_x[self.labels_km == i, 0], data_x[self.labels_km == i, 1], color=next(colour),
                        marker='.', linestyle='None')
        pyplot.plot(centroids[:,0], centroids[:,1], 'ko')
        pyplot.show()


    # K-means algorithm for a particular initialization of centroids
    def kmeans_algorithm(self, data_x, n_clusters, max_iterations, centroids, result_sse, result_labels):
        # Local variables
        n_instances = data_x.shape[0]
        n_features = data_x.shape[1]
        resta = np.zeros((n_instances, n_clusters))
        prev_SSE = 0

        # Until max_iterations, assign each data to its closest centroid and recompute centroids
        for iterations in range(0, max_iterations):
            new_centroids = np.zeros((n_clusters, n_features))
            m_instpercluster = [0] * n_clusters

            # Compute euclidean distance between the data points and the centroids
            for i in range(0, n_clusters):
                resta[:, i] = np.sum((data_x[:, :] - centroids[i, :]) ** 2, axis=1)

            # Compute SSE for that specific iteration
            SSE = np.sum(np.min(resta, axis=1))
            print('SSE in iteration ' + str(iterations) + ' is: ' + str(round(SSE,2)))

            # Assign each data to its closest centroid
            lista = np.argmin(resta, axis=1)

            # Recompute centroids
            for i in range(0, n_clusters):
                info = data_x[np.argwhere(lista == i).reshape(np.argwhere(lista == i).shape[0], ), :]
                new_centroids[i, :] = np.sum(info, axis=0)
                m_instpercluster[i] = np.sum(lista == i)
                centroids[i, :] = new_centroids[i, :] / m_instpercluster[i]

            if (self._has_converged(prev_SSE,SSE)):
                break
            else:
                prev_SSE = SSE

        print('SSE for specific initialization ' + ' --> ' + str(round(SSE,2))+'\n')
        result_sse.append(SSE)
        result_labels.append(lista)
        return result_sse, result_labels

    # Constructor
    def __init__(self, num_clusters, num_tries_init, max_iterations):
        self.num_clusters = num_clusters
        self.num_tries_init = num_tries_init
        self.max_iterations = max_iterations