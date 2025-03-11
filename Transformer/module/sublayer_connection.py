import tensorflow as tf
from tensorflow.keras.layers import Dropout


class sublayer_connection(tf.keras.layers.Layer):
    def call(self, inputs, sublayer, dropout = 0.2):
        inputs = tf.keras.layers.Add()([inputs, Dropout(dropout)(sublayer)])
        
        feature_shape = inputs.get_shape()[-1:]
        mean = tf.keras.backend.mean(inputs, [-1], keepdims = True)
        std = tf.keras.backend.std(inputs, [-1], keepdims = True)
        beta = tf.cast(tf.zeros(feature_shape), tf.float32)
        gamma = tf.cast(tf.ones(feature_shape), tf.float32)
        
        layer_norm = gamma *(inputs - mean) / (std + 1e-6) + beta
        return layer_norm
