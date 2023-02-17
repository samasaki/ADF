import numpy as np
import sys
sys.path.append("../")
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1' # need to install tensorflow-determinism

from adf_data.factory import DataFactory
from adf_model.tutorial_models import dnn
from adf_utils.utils import gpu_initialize, set_seed

def training(dataset, model_path, nb_epochs, batch_size,learning_rate):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    X, Y, input_shape, nb_classes, data_config = DataFactory.factory(dataset)


    model = dnn(input_shape, nb_classes, learning_rate)
    history = model.fit(X, Y, batch_size=batch_size, epochs=nb_epochs, shuffle=True)
    model.save(model_path + dataset + '/' + 'test.model.h5')
    
    accuracy = np.mean(history.history['categorical_accuracy'])
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))


def main(argv=None):
    gpu_initialize()
    set_seed()
    training(**argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='create a trained testing model')
    parser.add_argument('--dataset', type=str, default='census', help='the name of dataset')
    parser.add_argument('--model_path', type=str, default='../models/', help='the path for testing model')
    parser.add_argument('--nb_epochs', type=int, default=1000, help='Number of epochs to train model')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of training batches')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    argv = parser.parse_args()

    main(vars(argv))
