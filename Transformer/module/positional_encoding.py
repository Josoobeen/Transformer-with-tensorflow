import numpy as np
import tensorflow as tf

def positional_encoding(dim, sentence_length):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)], dtype = np.float32)
    encoded_vec[0::2] = np.sin(encoded_vec[0::2]) #2i에 사인함수
    encoded_vec[1::2] = np.cos(encoded_vec[1::2]) #2i + 1에 코사인 함수
    
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype = tf.float32)