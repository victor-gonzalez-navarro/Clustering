from sklearn.cluster import AgglomerativeClustering as Ag


class Agglomerative:
    labels_agg = None

    def agglomerative_method(self, data_x):
        print('\n'+'Computing clusters with Agglomerative Clustering')
        self.labels_agg = Ag(affinity=self.affinity, n_clusters=self.num_clusters, linkage=self.linkage).fit_predict(
            data_x)

    def __init__(self, num_clusters =2, affinity='euclidean', linkage='ward'):
        self.num_clusters = num_clusters
        self.affinity = affinity
        self.linkage = linkage
