from sklearn.cluster import AgglomerativeClustering as Ag
from eval_plot.evaluation import ploting_v


class Agglomerative:
    labels_agg = None


    # Main algorithm
    def agglomerative_method(self, data_x):
        print('\n'+'\033[1m'+'Computing clusters with Agglomerative Clustering algorithm...'+'\033[0m')

        # Compute Agglomerative Clustering algorithm with sklearn library
        self.labels_agg = Ag(affinity=self.affinity, n_clusters=self.num_clusters, linkage=self.linkage).fit_predict(
            data_x)

        # Scatter plot
        ploting_v(data_x, self.num_clusters, self.labels_agg)

        print('\033[1m'+'\nAccuracy:'+'\033[0m')


    # Constructor
    def __init__(self, num_clusters =2, affinity='euclidean', linkage='ward'):
        self.num_clusters = num_clusters
        self.affinity = affinity
        self.linkage = linkage
