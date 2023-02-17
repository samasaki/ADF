import numpy as np
import sys
sys.path.append("../")
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1' # need to install tensorflow-determinism

import tensorflow as tf
from tensorflow.python.platform import flags
from adf_data.census import census_data
from adf_data.bank import bank_data
from adf_data.credit import credit_data
from adf_model.tutorial_models import dnn
from adf_utils.utils import set_seed

FLAGS = flags.FLAGS

def gpu_initialize():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def training(dataset, model_path, nb_epochs, batch_size,learning_rate):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    X, Y, input_shape, nb_classes = data[dataset]()

    model = dnn(input_shape, nb_classes, learning_rate)
    history = model.fit(X, Y, batch_size=batch_size, epochs=nb_epochs, shuffle=True)
    model.save(model_path + dataset + '/' + 'test.model.h5')
    
    accuracy = np.mean(history.history['categorical_accuracy'])
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))


def main(argv=None):
    gpu_initialize()
    set_seed()
    training(dataset = FLAGS.dataset,
             model_path = FLAGS.model_path,
             nb_epochs=FLAGS.nb_epochs,
             batch_size=FLAGS.batch_size,
             learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_string("model_path", "../models/", "the name of path for saving model")
    flags.DEFINE_integer('nb_epochs', 1000, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

    main()