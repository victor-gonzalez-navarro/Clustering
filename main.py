import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
from preprocess import Preprocess
from scipy.io import arff
from sklearn import preprocessing as prp
from sklearn.cluster import AgglomerativeClustering as ag
from nltk.metrics.scores import accuracy
from matplotlib.pyplot import figure

def obtain_arffs(path):
   arffs_dic = {}
   for filename in os.listdir(path):
       if re.match('(.*).arff', filename):
           arffs_dic[filename.replace('.arff', '')] = arff.loadarff(path + filename)
   return arffs_dic

def main():
    # Reading and preprocess
    arffs_dic = obtain_arffs('./datasets/')
    dat1 = arffs_dic['grid']  # 'grid.arff' 'vehicle.arff'
    df1 = pd.DataFrame(dat1[0])  # pandas
    labels = df1['class'].values
    df1 = df1.drop('class', 1)
    data_x = df1.values  # numpy array

    load = Preprocess()
    load.preprocess_method(data_x)
    data_xy = np.concatenate((data_x,labels.reshape(len(data_x),1)),axis=1)

    ####################################################################################
    # HYPERPARAMETERS AGGLOMERATIVE CLUSTERING

    # HYPERPARAMETERS K-MEANS

    # HYPERPARAMETERS BISECTING K-MEANS

    # ...

    ####################################################################################

if __name__ == '__main__':
    main()