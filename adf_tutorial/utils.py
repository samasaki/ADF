import sys
sys.path.append("../")
from sklearn.cluster import KMeans
import joblib
import os
import tensorflow as tf
from tensorflow.python.platform import flags

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data

FLAGS = flags.FLAGS

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

def gradients(model, x, y=None):
    """
    Calculate gradients of the TF graph
    :param model: the TF model
    :param x: inputs
    :param y: labels
    :return: the gradients
    """
    tf_x = tf.Variable(x)
    with tf.GradientTape() as g:
        preds = model(tf_x)

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            preds_max = tf.reduce_max(preds, axis=1)
            labels = tf.cast(tf.equal(preds, preds_max), dtype=tf.float32)
        else:
            labels = tf.constant(y)

        loss = tf.losses.categorical_crossentropy(labels, preds)
    
    grads = g.gradient(loss, tf_x).numpy()

    return grads

def main(argv=None):
    cluster(dataset=FLAGS.dataset,
            cluster_num=FLAGS.clusters)

if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'name of datasets')
    flags.DEFINE_integer('clusters', 4, 'number of clusters')

    main()
