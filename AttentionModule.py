import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Softmax, Flatten


class AttentionModule(keras.layers.Layer):
    #
    def __init__(self):#, units=32, input_dim=32):
        super(AttentionModule, self).__init__()
        #u_init = tf.random_normal_initializer()
        # alternative: tf.zeros_initializer()
        #self.u = tf.Variable(
        #    initial_value=u_init(shape=(input_dim, units), dtype="float32"),
        #    trainable=True,
        #)

    def call(self, inputs, **kwargs):
        local_features, global_features = inputs

        # dimensionality mapping g to l
        global_feature_mapping = self.mapping_layer(global_features)

        # calculate compability scores
        compability_scores = self.compability_function(local_features, global_feature_mapping)

        # calculate attention
        attention = Softmax()(compability_scores)
        attention = Flatten()(attention)

        return attention

    def build(self, input_shape):
        local_feature_shape, global_feature_shape = input_shape
        self.mapping_layer = Dense(local_feature_shape[-1])

    # simple dot
    def compability_function(self, local_features, global_features):
        return K.batch_dot(local_features, global_features)
