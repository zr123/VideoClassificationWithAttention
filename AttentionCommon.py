import tensorflow as tf
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.activations import sigmoid, softmax


def sigmoid2d(x, name):
    shape = x.shape
    x = Flatten()(x)
    x = sigmoid(x)
    x = Reshape(shape[1:], name=name)(x)
    return x


def softmax2d(x, name):
    shape = x.shape
    x = Flatten()(x)
    x = softmax(x)
    x = Reshape(shape[1:], name=name)(x)
    return x


# pseudo-softmax: subtract by min value and divide by sum -> less sparse but similar properties as softmax
def pseudo_softmax2d(x, name):
    shape = x.shape
    x = Flatten()(x)
    a_min = tf.math.reduce_min(x, axis=-1, keepdims=True)
    a_sum = tf.math.reduce_sum(x - a_min, axis=-1, keepdims=True)
    x = (x - a_min) / a_sum
    x = Reshape(shape[1:], name=name)(x)
    return x
