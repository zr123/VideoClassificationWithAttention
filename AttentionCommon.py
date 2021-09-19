import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations


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
        lambda x: (x - tf.math.reduce_min(x, axis=-1, keepdims=True)) / tf.math.reduce_sum(x, axis=-1, keepdims=True)
    )(x)
    x = layers.Reshape(shape[1:], name=name)(x)
    return x
