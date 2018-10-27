import numpy as np
from eval_plot.evaluation import ploting_v


class Kmedoids:
    labels_kmedoids = None

    # Main algorithm
    def kmedoids_method(self, data_x):
        print('\n'+'\033[1m'+'Computing clusters with K-medoids algorithm...'+'\033[0m')

        # Local variables
        result_sse = []
        result_labels = []

        # Compute kmedoids for different initialization of the clusters
        for nm in range(0, self.num_tries_init):
            # Random initialization of the medoids
            centroids = data_x[np.random.randint(0, len(data_x) - 1, self.num_clusters), :]
            result_sse, result_labels = self.kmedoids_algorithm(data_x, self.num_clusters, self.max_iterations,
                                                               centroids, result_sse, result_labels)

        # Show the accuracy obtained with the best initialization
        print('\033[1m'+'Accuracy with initalization: '+str(np.argmin(result_sse))+' (the best one)'+'\033[0m')
        self.labels_kmedoids = result_labels[np.argmin(result_sse)]
        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(min(result_sse),

                                                                                       2)) + '\033[0m')

        # Scatter plot
        ploting_v(data_x, self.num_clusters, self.labels_kmedoids)

    # K-medoids algorithm for a particular initialization of medoids
    def kmedoids_algorithm(self, data_x, n_clusters, max_iterations, centroids, result_sse, result_labels):
        # Local variables
        n_instances = data_x.shape[0]
        resta = np.zeros((n_instances, n_clusters))

        # Until max_iterations, assign each data to its closest medoid and recompute medoids
        for iterations in range(0, max_iterations):

            # Compute euclidean distance between the data points and the centroids
            for i in range(0, n_clusters):
                resta[:, i] = np.sum((data_x[:, :] - centroids[i, :]) ** 2, axis=1)

            # Compute SSE for that specific iteration
            SSE = np.sum(np.min(resta, axis=1))
            print('SSE in iteration ' + str(iterations) + ' is: ' + str(SSE))

            # Assign each data to its closest medoid
            lista = np.argmin(resta, axis=1)  # put each instance to each cluster

            # Recompute centroids
            for k in range(0, n_clusters):
                bcx = sum(item == k for item in lista)  # number of instances in cluster k
                mat = np.float64(np.identity(bcx, bcx) * 5)
                whe = np.argwhere(lista == k)
                for i in range(0, bcx):  # to decide the medoid of cluster k
                    p1 = data_x[whe[i, 0],:]
                    for j in range(i + 1, bcx):
                        p2 = data_x[whe[j, 0], :]
                        mat[i, j] = np.sum((p1 - p2) ** 2)  # Euclidean Distance
                        mat[j, i] = mat[i, j]

                posicion = np.argmin(np.sum(mat, axis=1))
                centroids[k, :] = data_x[whe[posicion, 0], :]

        print('SSE for specific initialization ' + ' --> ' + str(round(SSE,2))+'\n')
        result_sse.append(SSE)
        result_labels.append(lista)
        return result_sse, result_labels

    # Constructor
    def __init__(self, num_clusters, num_tries_init, max_iterations):
        self.num_clusters = num_clusters
        self.num_tries_init = num_tries_init
        self.max_iterations = max_iterations
