import sys
sys.path.append("../")
import random
import os

import numpy as np
import tensorflow as tf
import joblib

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
