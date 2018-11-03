import numpy as np

from eval_plot.evaluation import ploting_v


class FuzzyCMeans:
    labels_fuzzyCM = None

    def __init__(self, n_clusters=2, m=2, eps=0.01, maxIter=100):
        self.maxIter = maxIter
        self.eps = eps
        self.n_clusters = n_clusters
        self.m = m

    def _init_membership_matrix(self, n_samples):
        '''
        Initialize the membership matrix randomly. Then, normalize it to sum one for each sample among clusters
        '''
        membership_matrix = []
        for sample in range(n_samples):
            c_randoms = [np.random.rand() for c in range(self.n_clusters)]
            norm_factor = sum(c_randoms)
            membership_matrix.append([memb_value / norm_factor for memb_value in c_randoms])
        return np.array(membership_matrix)

    def _calculate_center_vecs(self, memb_matrix, data):
        '''
        The function it represents is: c_j = (Σ_j_1^c(u_ij ^ m * x_i)) / (Σ_j_1^c(u_ij ^ m))
        '''
        centers = np.empty((self.n_clusters, data.shape[1]))
        for c in range(self.n_clusters):
            denominator = sum(memb_matrix[:, c] ** self.m)
            numerator = np.dot(memb_matrix.T[c, :] ** self.m, data)
            centers[c, :] = numerator / denominator

        return centers

    def _update_memb_matrix(self, data, centers):
        '''
        The function it represents is: c_j = 1 / (Σ_k_1^c (||x_i - c_j||/||x_i - c_k||) ^ (2 / (m - 1))
        '''
        memb_matrix = np.zeros((data.shape[0], self.n_clusters))
        for i in range(data.shape[0]):
            dist_to_clusters = [np.linalg.norm(data[i, :] - centers[j, :]) for j in range(self.n_clusters)]
            exp_factor = 2 / (self.m - 1)
            for j in range(self.n_clusters):
                denominator = sum(
                    [(dist_to_clusters[j] / dist_to_clusters[k]) ** exp_factor for k in range(self.n_clusters)])
                memb_matrix[i, j] = 1 / denominator

        return memb_matrix

    def _get_clusters(self, memb_matrix):
        '''
        Returns the labels array given a membership matrix by selecting the cluster with maximum memb_value per sample
        '''
        cluster_labels = []
        for i in range(memb_matrix.shape[0]):
            idx = max(range(self.n_clusters), key=lambda s: memb_matrix[i, s])
            cluster_labels.append(idx)
        return np.array(cluster_labels)

    def _get_sse_per_cluster(self, data, centers, labels):
        '''
        Returns the SSE per cluster
        '''
        sse = np.zeros((centers.shape[0], 1))
        for i in range(data.shape[0]):
            sse[labels[i], 0] += np.linalg.norm(data[i, :] - centers[labels[i], :])**2

        return sse

    def fit(self, data):
        print('\n' + '\033[1m' + 'Computing clusters with Fuzzy C-means algorithm...' + '\033[0m')

        n_samples = data.shape[0]

        # Initialize U matrix (membership funtion matrix)
        memb_matrix = self._init_membership_matrix(n_samples)
        iteration = 0
        converged = False

        while (not converged and iteration < self.maxIter):
            # At k-step: calculate the center vectors C=[c_j] with U
            centers = self._calculate_center_vecs(memb_matrix, data)
            # Update U with the new center vectors C
            new_m_matrix = self._update_memb_matrix(data, centers)
            if (np.linalg.norm(new_m_matrix - memb_matrix) < self.eps):
                converged = True
                self.labels_fuzzyCM = self._get_clusters(memb_matrix)
            else:
                iteration += 1
                memb_matrix = np.copy(new_m_matrix)

        print('It has converged at the ' + str(iteration) + 'th iteration.')

        total_see = np.sum(self._get_sse_per_cluster(data, centers, self.labels_fuzzyCM))

        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(total_see, 2)) + '\033[0m')

        # Scatter plot
        #ploting_v(data, self.n_clusters, self.labels_fuzzyCM)


