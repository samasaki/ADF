import sys
sys.path.append("../")
import random
import os

import numpy as np
import tensorflow as tf
import joblib

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

def load_cluster(dataset):
    model_path = '../clusters/' + dataset + '.pkl'
    clf = joblib.load(model_path)
    return clf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
