import os
import re

import pandas as pd
from scipy.io import arff

from preprocess_b import Preprocess_b
from evaluation import evaluate
from agglomerative import Agglomerative
from kmeans import Kmeans


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
    num_clusters = 2
    affinity = 'euclidean'
    linkage = 'ward'  # ['ward', 'complete', 'average']

    tst1 = Agglomerative(num_clusters,affinity,linkage)
    tst1.agglomerative_method(data_x)
    evaluate(tst1.labels_agg, groundtruth_labels)


# --------------------------------------------------------------------------------------------------- K-means Clustering
def tester_kmeans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 4
    num_tries_init = 2
    max_iterations = 4

    tst2 = Kmeans(num_clusters, num_tries_init, max_iterations)
    tst2.kmeans_method(data_x)
    evaluate(tst2.labels_km, groundtruth_labels)


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    arffs_dic = obtain_arffs('./datasets/')

    # Extract an specific database
    dat1 = arffs_dic['grid']
    df1 = pd.DataFrame(dat1[0])  # original data in pandas dataframe
    groundtruth_labels = df1['class'].values  # original labels in a numpy array
    df1 = df1.drop('class',1)
    data1 = df1.values  # original data in a numpy array without labels
    load = Preprocess_b()
    data_x = load.preprocess_methodb(data1)

    # Clustering algorithms
    mth = 2
    if mth == 1:
        tester_agglomerative(data_x,groundtruth_labels)
    elif mth == 2:
        tester_kmeans(data_x, groundtruth_labels)


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()