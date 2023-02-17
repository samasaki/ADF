import sys
sys.path.append("../")
import random
import argparse

import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
import tensorflow as tf

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data

datasets_dict = {'census':census_data, 'credit':credit_data, 'bank': bank_data}

def cluster(dataset, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset: the name of dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    if os.path.exists('../clusters/' + dataset + '.pkl'):
        clf = joblib.load('../clusters/' + dataset + '.pkl')
    else:
        X, Y, input_shape, nb_classes = datasets_dict[dataset]()
        clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
        joblib.dump(clf , '../clusters/' + dataset + '.pkl')
    return clf

def gpu_initialize():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def set_seed(seed=1234):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def main(argv=None):
    cluster(**argv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='execute symbolic generation')
    parser.add_argument('--dataset', type=str, default='census', help='the name of dataset')
    parser.add_argument('--cluster_num', type=int, default=4, help='number of clusters')
    argv = parser.parse_args()

    main(vars(argv))
