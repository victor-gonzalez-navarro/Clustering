import numpy as np

from eval_plot.evaluation import ploting_v

class Clarans:
    labels_clarans = None

    # Main algorithm
    def clarans_method(self, data_x):
        print('\n' + '\033[1m' + 'Computing clusters with CLARANS algorithm...' + '\033[0m')
        listSSE = []
        listLabels = []
        listcentro = []

        # Compute CLARANS for different initialization of the medoids (number of local minima you want to explore)
        for numi in range(self.numlocal):
            data_whereclusters = np.round(np.linspace(numi, len(data_x) -1- numi, self.num_clusters)).astype(int)
            centroids = data_x[data_whereclusters, :]
            listSSE, listLabels, listcentro = self.clarans_algorithm(data_x, self.num_clusters,
                self.max_iterations, centroids, data_whereclusters, self.max_neighbour, listSSE, listLabels, listcentro)

        # Show the accuracy obtained with the best initialization
        print('\n\033[1m'+'Accuracy with initalization: '+str(np.argmin(listSSE))+' (the best one)'+'\033[0m')
        self.labels_clarans = listLabels[np.argmin(listSSE)]
        print('The SSE (sum of squared errors) is: ' + '\033[1m' + '\033[94m' + str(round(min(listSSE), 2)) + '\033[0m')

        # Scatter plot
        ploting_v(data_x, self.num_clusters, self.labels_clarans)

    # CLARANS algorithm for a particular initialization of medoids. This algorithm has been done following the cases
    # of the original paper so as to compute TCMP
    def clarans_algorithm(self, data_x, n_clusters, max_iterations, centroids, data_whereclusters,
                          maxneighbor, listSSE, listLabels, listcentro):
        # Local variables
        n_instances = data_x.shape[0]

        # Initialization
        resta = np.zeros((n_instances, n_clusters))
        fin = 0
        iterations = 0
        numplots = 0

        # Until max_iterations or min(TCMP)>0 or I have already search enough local minimums, assign each data to its
        # closest medoid and recompute medoid
        while (iterations < max_iterations) and (fin == 0) and (numplots < maxneighbor):
            # Local variables to stop iterating ech of the while loops
            fin = 1
            endfor1and2 = False
            # Decide cluster for each instance
            for i in range(0, n_clusters):
                resta[:, i] = np.sum((data_x[:, :] - centroids[i, :]) ** 2, axis=1)
            lista = np.argmin(resta, axis=1)  # put each instance to each cluster


            # Compute matrix Om*Op (with all possible p and m (m are clusters))
            # Om to denote a current medoid that is to be replaced
            ki1 = 0
            while (ki1 < (n_clusters)) and (endfor1and2 == False):
                # Main difference with PAM: find random neighbour
                k1 = np.round(np.random.rand(1)[0] * (n_clusters - 1)).astype(int)
                Om = centroids[k1, :]
                # Op to denote the new medoid to replace Om
                ki2 = 0
                while (ki2 < len(data_x)) and (endfor1and2 == False):
                    k2 = np.round(np.random.rand(1)[0] * (len(data_x) - 1)).astype(int)
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
                        # Main difference with PAM, centroids are recomputed everytime TCMP is negative
                        if TCMP < 0:
                            centroids[k1, :] = data_x[k2,:]
                            data_whereclusters[k1] = k2
                            endfor1and2 = True
                            fin = 0
                            numplots = numplots + 1

                    ki2 = ki2 + 1
                ki1 = ki1 + 1

            iterations = iterations + 1

        # Compute euclidean distance between data points and medoids
        for i in range(0, n_clusters):
            resta[:, i] = np.sum((data_x[:, :] - centroids[i, :]) ** 2, axis=1)

        # Compute SSE
        SSE = np.sum(np.min(resta, axis=1))

        print('SSE (local minimum) for specific initialization ' + ' --> ' + str(round(SSE,2)))

        # Save SSE, labels for each datapoint and centroids
        listSSE.append(SSE)
        listLabels.append(lista)
        listcentro.append(centroids)
        return listSSE, listLabels, listcentro

    # Constructor
    def __init__(self, num_clusters, numlocal, max_iterations, max_neighbour):
        self.num_clusters = num_clusters
        self.numlocal = numlocal
        self.max_iterations = max_iterations
        self.max_neighbour = max_neighbour
