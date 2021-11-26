import tensorflow as tf
from tensorflow.keras import layers, activations


def softmax2d(x, name):
    shape = x.shape
    x = layers.Flatten()(x)
    x = activations.softmax(x)
    x = layers.Reshape(shape[1:], name=name)(x)
    return x


# pseudo-softmax: subtract by min value and divide by sum -> less sparse but similar properties as softmax
def pseudo_softmax2d(x, name):
    shape = x.shape
    x = layers.Flatten()(x)
    # x = (x - min(x)) / sum(x)
    x = layers.Lambda(
        lambda l: (l - tf.math.reduce_min(l, axis=-1, keepdims=True)) / tf.math.reduce_sum(l, axis=-1, keepdims=True)
    )(x)
    x = layers.Reshape(shape[1:], name=name)(x)
    return x
