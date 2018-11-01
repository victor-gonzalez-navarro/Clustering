import os
import re

import pandas as pd
from scipy.io import arff

from clust_alg.agglomerative import Agglomerative
from clust_alg.kmeans import Kmeans
from clust_alg.bisectingKmeans import BisectingKmeans
from clust_alg.kmedoids import Kmedoids
from clust_alg.kmedoids_b import Kmedoids_b
from clust_alg.pam import Pam
from clust_alg.clarans import Clarans
from clust_alg.fuzzyCMeans import FuzzyCMeans
from eval_plot.evaluation import evaluate
from preproc.preprocess_b import Preprocess_b


# ------------------------------------------------------------------------------------------------------- Read databases
def obtain_arffs(path):
    # Read all the databases
    arffs_dic = {}
    for filename in os.listdir(path):
        if re.match('(.*).arff', filename):
            arffs_dic[filename.replace('.arff', '')] = arff.loadarff(path + filename)
    return arffs_dic


# --------------------------------------------------------------------------------------------- Agglomerative Clustering
def tester_agglomerative(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 2        # Number of clusters
    affinity = 'euclidean'
    linkage = 'ward'        # ['ward', 'complete', 'average']

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m'+'\nNumber of clusters: '+str(
        num_clusters)+'\nAffinity distance: '+affinity+'\nLinkage: '+linkage)

    tst1 = Agglomerative(num_clusters,affinity,linkage)
    tst1.agglomerative_method(data_x)
    evaluate(tst1.labels_agg, groundtruth_labels)


# -------------------------------------------------------------------------------------------------------------- K-means
def tester_kmeans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 4        # Number of clusters
    num_tries_init = 2      # Number of different initializations of the centroids
    max_iterations = 8      # Number of iterations for each initialization

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m'+'\nNumber of clusters: '+str(
        num_clusters)+'\nNumber of different initilizations: '+str(num_tries_init)+'\nMaximum number of iterations '
                                                                            'per initialization: '+str(max_iterations))

    tst2 = Kmeans(num_clusters, num_tries_init, max_iterations)
    tst2.kmeans_method(data_x)
    evaluate(tst2.labels_km, groundtruth_labels)


# ---------------------------------------------------------------------------------------------------- Bisecting K-means
def tester_bisectingKmeans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 2        # Number of clusters
    max_iterations = 3      # Number of iterations for each initialization
    max_bisect = 2          # For example, 2 will indicate that two clusters clusters will be split

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m'+'\nNumber of clusters: '+str(
        num_clusters)+'\nNumber of clusters to split: '+str(max_bisect)+'\nMaximum number of iterations '
                                                                            'per initialization: '+str(max_iterations))

    tst3 = BisectingKmeans(num_clusters, max_iterations, max_bisect)
    tst3.bisectkmeans_method(data_x)
    evaluate(tst3.labels_bisectkm, groundtruth_labels)


# ------------------------------------------------------------------------------------------------------------ K-medoids
def tester_kmedoids(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 3        # Number of clusters
    num_tries_init = 2      # Number of different initializations of the centroids
    max_iterations = 4      # Number of iterations for each initialization

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m' + '\nNumber of clusters: ' + str(
        num_clusters) + '\nNumber of different initilizations: ' + str(
        num_tries_init) + '\nMaximum number of iterations per initialization: ' + str(max_iterations))

    tst3 = Kmedoids(num_clusters, num_tries_init, max_iterations)
    tst3.kmedoids_method(data_x)
    evaluate(tst3.labels_kmedoids, groundtruth_labels)


# ------------------------------------------------------------------------------------------------------------ K-medoids
def tester_kmedoids_b(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 3        # Number of clusters
    max_iterations = 6      # Number of iterations for each initialization

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m' + '\nNumber of clusters: ' + str(
        num_clusters) + '\nMaximum number of iterations: ' + str(max_iterations))

    tst3 = Kmedoids_b(num_clusters, max_iterations)
    tst3.fit(data_x)
    evaluate(tst3.labels_, groundtruth_labels)


# ------------------------------------------------------------------------------------------------------------------ PAM
def tester_pam(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 2        # Number of clusters
    max_iterations = 3      # Number of iterations for each initialization

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m' + '\nNumber of clusters: ' + str(
        num_clusters) + '\nMaximum number of iterations: ' + str(max_iterations))

    tst4 = Pam(num_clusters, max_iterations)
    tst4.pam_method(data_x)
    evaluate(tst4.labels_pam, groundtruth_labels)


# -------------------------------------------------------------------------------------------------------------- CLARANS
def tester_clarans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 4        # Number of clusters
    numlocal = 2            # Number of local minimum I want to search (parameter of CLARANS algorithm)
    max_iterations = 8      # Number of iterations for each initialization
    max_neighbor = 5        # If this number is very big, CLARANS becomes more similar to PAM

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m' + '\nNumber of clusters: ' + str(
        num_clusters) + '\nNumber of different initilizations: ' + str(
        numlocal) + '\nMaximum number of iterations per initialization: ' + str(max_iterations) +'\nHyperparameter '
                                                            'max_neighbours in CLARANS\' algorithm: '+str(max_neighbor))

    tst5 = Clarans(num_clusters, numlocal, max_iterations, max_neighbor)
    tst5.clarans_method(data_x)
    evaluate(tst5.labels_clarans, groundtruth_labels)

# -------------------------------------------------------------------------------------------------------- FUZZY C-MEANS
def tester_fuzzyCmeans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 3        # Number of clusters
    m = 2                   # The Fuzzy parameter can be any real number greater than 1
    eps = 0.01              # Threshold of convergence
    max_iterations = 100    # Max Number of iterations until convergence

    print('\n' + '\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m' + '\nNumber of clusters: ' + str(
        num_clusters) + '\nM fuzzy parameter: ' + str(m) + '\nEps: ' + str(eps) +
          '\nMaximum number of iterations: ' + str(max_iterations))

    tst6 = FuzzyCMeans(num_clusters, m, eps, max_iterations)
    tst6.fit(data_x)
    evaluate(tst6.labels_fuzzyCM, groundtruth_labels)


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the databases...' + '\033[0m')
    arffs_dic = obtain_arffs('./datasets/')

    # Extract an specific database
    dat1 = arffs_dic['iris']
    df1 = pd.DataFrame(dat1[0])  # original data in pandas dataframe
    groundtruth_labels = df1['class'].values  # original labels in a numpy array
    df1 = df1.drop('class',1)
    data1 = df1.values  # original data in a numpy array without labels
    load = Preprocess_b()
    data_x = load.preprocess_methodb(data1)

    # Clustering algorithms
    print('\n\033[1m'+'Chose which clustering algorithm you want to test: '+'\033[0m')
    print('OPTIONS: \n1) Agglomerative Clustering\n2) K-means \n3) Bisecting K-means \n4) K-medoids (implemented as '
          'K-means) \n5) Partitioning Around Medoids (PAM) \n6) CLARANS \n7) Fuzzy C-Means')
    inputUser = int(input('Introduce the number and then press enter: '))
    mth = inputUser

    if mth == 1:
        tester_agglomerative(data_x,groundtruth_labels)
    elif mth == 2:
        tester_kmeans(data_x, groundtruth_labels)
    elif mth == 3:
        tester_bisectingKmeans(data_x, groundtruth_labels)
    elif mth == 4:
        tester_kmedoids_b(data_x, groundtruth_labels)
    elif mth == 5:
        tester_pam(data_x, groundtruth_labels)
    elif mth == 6:
        tester_clarans(data_x, groundtruth_labels)
    elif mth == 7:
        tester_fuzzyCmeans(data_x, groundtruth_labels)
    else:
        print('The number introduced is not accpected')


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()