

import tensorflow as tf
import numpy as np
from tensorflow import keras

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500, verbose=0)

print(model.predict([10.0]))

def my_huber_loss_self(y_pred, y_true):
    #https: // en.wikipedia.org / wiki / Huber_loss
    threshold = 1
    error = y_pred - y_true
    if tf.abs(error) <= threshold:
        loss = tf.square(error)/2.0
    else:
        loss = threshold*(tf.abs(error)-(0.5*threshold))
    return loss

def my_huber_loss(y_pred, y_true):
    # This is the standardized code from coursera, above function is how I implemented.
    threshold = 1
    error = y_pred - y_true
    is_small_error = tf.abs(error)<threshold
    small_error_loss = tf.square(error)/2
    big_error_loss = threshold * (tf.abs(error)-(0.5*threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=my_huber_loss)
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.0]))











