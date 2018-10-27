import numpy as np

from eval_plot.evaluation import ploting_v


class Pam:
    labels_pam = None

    # Main algorithm
    def pam_method(self, data_x):
        print('\n'+'\033[1m'+'Computing clusters with PAM algorithm... (this algorithm has a computational '
                             'complexity of K(N-K)^2)'+'\033[0m')

        # 1 iteration of PAM (we do not initialize with other medoids due to the computational complexity)
        data_whereclusters = np.round(np.linspace(0, len(data_x) - 1, self.num_clusters)).astype(int)
        centroids = data_x[data_whereclusters, :]
        self.labels_pam, result_sse = self.pam_algorithm(data_x, self.num_clusters, self.max_iterations, centroids,
                                           data_whereclusters)

        # Accuracy
        print('\n\033[1m'+'Accuracy: '+'\033[0m')
        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(result_sse,3)) + '\033[0m')

        # Scatter plot
        ploting_v(data_x, self.num_clusters, self.labels_pam)

    # Pam algorithm for a particular initialization of medoids. This algorithm has been done following the cases
    # of the original paper of CLARANS so as to compute TCMP
    def pam_algorithm(self, data_x, n_clusters, max_iterations, centroids, data_whereclusters):
        # Local variables
        n_instances = data_x.shape[0]

        # Initialization
        resta = np.zeros((n_instances, n_clusters))
        fin = 0
        iterations = 0

        # Until max_iterations or min(TCMP)>0, assign each data to its closest medoid and recompute medoid
        while (iterations < max_iterations) and (fin == 0):
            matrix = np.ones((n_clusters, len(data_x))) * 10000
            # Decide cluster for each instance
            for i in range(0, n_clusters):
                resta[:, i] = np.sum((data_x[:, :] - centroids[i, :]) ** 2, axis=1)

            # Compute SSE for that specific iteration
            SSE = np.sum(np.min(resta, axis=1))
            print('SSE in iteration ' + str(iterations) + ' is: ' + str(round(SSE,2)))

            # Assign each data to its closest centroid
            lista = np.argmin(resta, axis=1)  # put each instance to each cluster

            # Compute matrix Om*Op (with all possible p and m (m are clusters))
            # Om to denote a current medoid that is to be replaced
            for k1 in range(n_clusters):
                Om = centroids[k1, :]
                # Op to denote the new medoid to replace Om
                for k2 in range(len(data_x)):
                    if (k2 not in data_whereclusters):
                        Op = data_x[k2, :]
                        # Change centroid k1 by data k2
                        # centroids_mod[k1,:] = data1[k2,0:N_features]
                        TCMP = 0
                        # For all Oj I need to compute Cmpj
                        for k3 in range(len(data_x)):
                            Oj = data_x[k3, :]
                            dist1 = np.sum((Oj - Op) ** 2)
                            dist2 = np.sum((Oj - Om) ** 2)
                            if (k3 not in data_whereclusters) and (k3 != k2):
                                # Case 1 and 2: Oj currently belongs to the cluster represented by Om
                                if lista[k3] == k1:
                                    ttt = []
                                    for tttid in range(n_clusters):
                                        if (centroids[tttid, 0] != Om[0]) and (centroids[tttid, 1] != Om[1]):
                                            ttt.append(np.sum((Oj - centroids[tttid, :]) ** 2))
                                        else:
                                            ttt.append(1000)
                                    Oj2 = centroids[ttt.index(min(ttt)), :]
                                    dist3 = np.sum((Oj - Oj2) ** 2)
                                    # Case 2: Oj is less similar to Oj2 than to Op
                                    if dist1 < dist3:
                                        # d(Oj;Op) - d(Oj;Om)
                                        CMP = dist1 - dist2
                                    # Case 1: Oj is more similar to Oj2 than to Op
                                    else:
                                        # d(Oj;Oj2) - d(Oj;Om)
                                        CMP = dist3 - dist2

                                # Case 3 and 4: Oj currently belongs to a cluster other than the one represented by Om
                                else:
                                    Oj2 = centroids[lista[k3], :]
                                    # Case 3: Oj be more similar to Oj2 than to Op
                                    dist4 = np.sum((Oj - Oj2) ** 2)
                                    if dist1 > dist4:
                                        CMP = 0
                                    # Case 4: Oj is less similar to Oj2 than to Op
                                    else:
                                        CMP = dist1 - dist4

                                TCMP = TCMP + CMP

                        matrix[k1, k2] = TCMP

            print('Minimum (TCMP) = ' + str(round(np.min(matrix),3)))
            if np.min(matrix) >= 0:
                fin = 1
            rowmin, colmin = np.unravel_index(matrix.argmin(), matrix.shape)

            # Recompute new medoid
            centroids[rowmin, :] = data_x[colmin, :]
            data_whereclusters[rowmin] = colmin
            iterations = iterations + 1

        # Compute euclidean distance between data points and medoids
        for i in range(0, n_clusters):
            resta[:, i] = np.sum((data_x[:,:] - centroids[i, :]) ** 2, axis=1)

        # Assign each data to its closest centroid
        lista = np.argmin(resta, axis=1)  # put each instance to each cluster
        return lista, SSE

    # Constructor
    def __init__(self, num_clusters, max_iterations):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
