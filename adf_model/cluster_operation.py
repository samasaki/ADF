import sys
sys.path.append("../")
import argparse

from sklearn.cluster import KMeans
import joblib

from adf_data.factory import DataFactory

def make_cluster(dataset, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset: the name of dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    X, Y, input_shape, nb_classes, data_config = DataFactory.factory(dataset)
    clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
    joblib.dump(clf , '../clusters/' + dataset + '.pkl')

def main(argv=None):
    make_cluster(**argv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='execute symbolic generation')
    parser.add_argument('--dataset', type=str, default='census', help='the name of dataset')
    parser.add_argument('--cluster_num', type=int, default=4, help='number of clusters')
    argv = parser.parse_args()

    main(vars(argv))
