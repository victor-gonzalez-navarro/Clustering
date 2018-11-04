import numpy as np
import random
from eval_plot.evaluation import ploting_v

class Kmedoids_b:
    labels_ = None

    def __init__(self, n_clusters, num_tries_init, max_iterations):
        self.maxIter = max_iterations
        self.n_clusters = n_clusters
        self.num_tries_init = num_tries_init

    def _sse(self, a, b):
        return sum([(xi - xj) ** 2 for xi, xj in zip(a, b)])    # squared 2-norm

    def _obtain_distance_matrix(self, points, d_metric):
        d_matrix = np.full((points.shape[0], points.shape[0]), np.inf)
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                if (d_matrix[i, j] == np.inf):
                    d_matrix[i, j] = d_metric(points[i], points[j])  # Compute the distance using the input metric
                    d_matrix[j, i] = d_matrix[i, j]

        return d_matrix

    def _select_random_medoids(self, valid_idxs, invalid_idxs, n_medoids):
        idxs = []
        for i in range(n_medoids):
            idx = random.choice(valid_idxs)     # Select a random medoid that is not in the invalid_idx list
            while (idx in invalid_idxs):
                idx = random.choice(valid_idxs)
            idxs.append(idx)
            invalid_idxs.append(idx)
        return np.array(sorted(idxs))

    def _recompute_medoids(self, d_matrix, invalid_idxs, labels, costs, medoids):
        for label in range(len(medoids)):
            rs = np.where(labels == label)[0]               # Points corresponding to this cluster
            costes = np.sum(d_matrix[rs][:, rs], axis=0)    # Sum of the distances
            medoids[label] = rs[np.argmin(costes)]          # Select the one that minimize the costs
            costs[label] = min(costes)                      # Update the cost of this cluster
            invalid_idxs.append(medoids[label])             # Add the current medoid to the invalid_idx for future

    def _make_clusters(self, d_matrix, medoids):
        labels = np.zeros((d_matrix.shape[0], 1), dtype=int)
        c_costs = np.zeros((len(medoids), 1))

        for i in range(d_matrix.shape[0]):
            labels[i] = np.argmin(d_matrix[i, medoids])            # The medoid that is closer to the current data point
            c_costs[labels[i]] += d_matrix[i, medoids[labels[i]]]  # Add the current distance to the cost of this cluster

        return labels, c_costs

    def kmedoids_algorithm(self, data, d_matrix, result_sse, result_labels):
        converged = False
        iteration = 0
        # List for the medoids that have already been chosen
        invalid_medoids = []

        # Select random medoids from the dataset
        medoids = self._select_random_medoids(range(data.shape[0]), invalid_medoids, self.n_clusters)

        # Cluster the data according to the previous medoids
        labels_, c_costs = self._make_clusters(d_matrix, medoids)

        # Recompute the medoids for the clusters created
        self._recompute_medoids(d_matrix, invalid_medoids, labels_, c_costs, medoids)

        # Repeat the update steps while the clustering has not converged yet
        while (not converged):
            ### Possible approach:
            ### Select cluster that has the bigger sse and ramdomly change the medoid
            ## c_with_max_sse = np.argmax(c_costs)
            ## medoids[c_with_max_sse] = self._select_random_medoids(np.where(self.labels_ == c_with_max_sse)[0], invalid_medoids, 1)

            new_labels, new_costs = self._make_clusters(d_matrix, medoids)
            self._recompute_medoids(d_matrix, invalid_medoids, new_labels, new_costs, medoids)

            # It totally converges when the SSE does not decrease
            if (np.sum(c_costs) > np.sum(new_costs) and iteration < self.maxIter):
                labels_ = np.copy(new_labels)
                c_costs = np.copy(new_costs)
                iteration += 1
            else:
                converged = True
                # ploting_v(data, self.n_clusters, self.labels_)

        print('SSE for specific initialization ' + ' --> ' + str(round(np.sum(c_costs), 2)) + '\n')
        result_sse.append(np.sum(c_costs))
        result_labels.append(labels_)
        return result_sse, result_labels

    def fit(self, data):
        print('\n' + '\033[1m' + 'Computing clusters with K-Medoids algorithm...' + '\033[0m')

        # Obtain the distance between each point in the dataset
        d_matrix = self._obtain_distance_matrix(data, self._sse)

        result_labels = []
        result_sse = []

        # Compute kmeans for different initialization of the clusters
        for nm in range(0, self.num_tries_init):
            result_sse, result_labels = self.kmedoids_algorithm(data, d_matrix, result_sse, result_labels)

        print('\033[1m' + 'Accuracy with initalization: ' + str(np.argmin(result_sse)) + ' (the best one)' + '\033[0m')
        self.labels_ = np.array(result_labels[np.argmin(result_sse)]).reshape(data.shape[0],)
        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(min(result_sse),
                                                                                          2)) + '\033[0m')



