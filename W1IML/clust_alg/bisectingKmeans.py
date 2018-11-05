import numpy as np

from eval_plot.evaluation import ploting_v


class BisectingKmeans:
    labels_bisectkm = None

    # Constructor
    def __init__(self, num_clusters, max_iterations, max_bisect):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.max_bisect = max_bisect

    # Main algorithm
    def bisectkmeans_method(self, data_x):
        print('\n'+'\033[1m'+'Computing clusters with Bisecting K-means algorithm...'+'\033[0m')

        # Local variable
        m_instpercluster = [0]  # Local variable
        self.labels_bisectkm = np.zeros((len(data_x), 1))
        centroids = data_x[np.random.randint(0, len(data_x) - 1, self.num_clusters)]  #Initialize centroids
        cluster_split = np.argmin(m_instpercluster)
        n_features = data_x.shape[1]

        # Start a first iteration of kmeans without splitting any centroid
        self.labels_bisectkm, m_instpercluster, result_sse = self.bisectkmeans_algorithm(data_x, self.num_clusters,
                                                                             self.max_iterations, centroids,
                                                                             self.labels_bisectkm, cluster_split)
        # Split a centroid and apply kmeans to the cluster
        for num_bisec in range(self.max_bisect):
            print('----One centroid has been duplicated----')
            cluster_split = np.argmin(m_instpercluster)
            self.num_clusters = self.num_clusters + 1
            centroids = np.concatenate((centroids, centroids[cluster_split, :].reshape(1, n_features) + 0.01), axis=0)
            self.labels_bisectkm, m_instpercluster, result_sse = self.bisectkmeans_algorithm(data_x, self.num_clusters,
                                                                                 self.max_iterations, centroids,
                                                                                 self.labels_bisectkm, cluster_split)


        # Accuracy for the last choice of number of centroids
        print('\n\033[1m'+'Accuracy: '+'\033[0m')
        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(result_sse,3)) + '\033[0m')

        # Scatter plot
        # ploting_v(data_x, self.num_clusters, self.labels_bisectkm)

    # Bisecting K-means algorithm for a particular initialization of centroids
    def bisectkmeans_algorithm(self, data_x, n_clusters, max_iterations, centroids, lista, cluster_split):

        # Local variables
        n_instances = data_x.shape[0]
        n_features = data_x.shape[1]
        resta = np.zeros((n_instances, n_clusters))
        listant = np.copy(lista)

        # Apply kmeans algorithm with max_iterations
        for iterations in range(0, max_iterations):
            new_centroids = np.zeros((n_clusters, n_features))
            m_instpercluster = [0] * n_clusters

            fil = np.argwhere(listant[:, 0] == cluster_split)  # shape = (N_instances,1)

            # Compute distance between all data points and all centroids
            for i in range(0, n_clusters):
                resta[fil[:, 0], i] = np.sum((data_x[fil[:, 0], 0:n_features] - centroids[i, :]) ** 2, axis=1)

            # Compute SSE for that specific iteration
            SSE = np.sum(np.min(resta, axis=1))
            print('SSE in iteration ' + str(iterations) + ' is: ' + str(round(SSE, 2)))

            # Split a cluster (cluster_split) a centroid by adding one next to it
            conc = np.concatenate(
                (resta[fil[:, 0], cluster_split].reshape(len(fil), 1), resta[fil[:, 0], -1].reshape(len(fil), 1)),
                axis=1)  # shape = (N_instances,2)
            res = np.argmin(conc, axis=1)  # shape = (N_instances,)
            for item in range(len(res)):
                if res[item] == 0:
                    res[item] = cluster_split
                else:
                    res[item] = (n_clusters - 1)

            lista[fil[:, 0], :] = res.reshape(len(res), 1)  # shape = (N_instances,1)

            # Recompute centroids for that specific iteration
            for i in [cluster_split, n_clusters - 1]:
                info = data_x[np.argwhere(lista[:, 0] == i).reshape(np.argwhere(lista[:, 0] == i).shape[0], ),
                       0:n_features]  # shape = (k,2)
                new_centroids[i, :] = np.sum(info, axis=0)
                m_instpercluster[i] = np.sum(lista == i)
                centroids[i, :] = new_centroids[i, :] / m_instpercluster[i]

        return lista, m_instpercluster, SSE
