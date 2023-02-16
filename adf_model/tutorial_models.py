import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU, Softmax

def dnn(input_shape=(13,), nb_classes=2, learning_rate=0.01):
    model = Sequential([
        Dense(64, input_shape=input_shape, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(nb_classes, activation='softmax')
    ], name='dnn_tutorial')

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), # TF2.10+ needs 'legacy' for a while
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    
    return model
