import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Softmax, Flatten, Reshape


class AttentionModule(keras.layers.Layer):
    """Attention module from LEARN TO PAY ATTENTION by Jetley et al.

      Arguments:
        compatibility_func: A string,
          one of `weighted` (default) or `dot`.

    """
    def __init__(self, compatibility_func="weighted"):
        super(AttentionModule, self).__init__()
        assert compatibility_func in ["weighted", "dot"], "Unexpected compatibility_func argument."
        self.compatibility_func = compatibility_func

    def call(self, inputs, **kwargs):
        local_features, global_features = inputs

        # dimensionality mapping g to l
        global_feature_mapping = self.mapping_layer(global_features)

        # calculate compatibility scores
        compatibility_scores = self.compatibility_function(local_features, global_feature_mapping)

        # calculate attention
        compatibility_scores = Flatten()(compatibility_scores)
        attention = Softmax()(compatibility_scores)
        attention = Reshape(self.shape)(attention)
        return attention

    def build(self, input_shape):
        local_feature_shape, global_feature_shape = input_shape
        # mapping layer, to make sure global feature shape and local feature shape is the same
        self.shape = local_feature_shape[1:-1]
        self.mapping_layer = Dense(local_feature_shape[-1])
        if self.compatibility_func == "weighted":
            self.u = self.add_weight(name='u', shape=(local_feature_shape[-1],), initializer="random_normal", trainable=True)

    def get_config(self):
        config = super(AttentionModule, self).get_config()
        config.update({
            'compatibility_func':
                self.compatibility_func
        })
        return config

    def compatibility_function(self, local_features, global_features):
        if self.compatibility_func == "weighted":
            return self.weighted_compatibility_function(local_features, global_features)
        elif self.compatibility_func == "dot":
            return self.simple_dot_compatibility_function(local_features, global_features)
        else:
            raise Exception("Unexpected compatibility function: " + self.compability_func)

    # simple dot
    def simple_dot_compatibility_function(self, local_features, global_features):
        return K.batch_dot(local_features, global_features)

    # weighted dot product
    def weighted_compatibility_function(self, local_features, global_features):
        # expand global_feateurs from (batch_size, n) to (batch_size, 1, 1, n)
        # so the addition is broadcast across all spatial features
        expanded_g = tf.expand_dims(tf.expand_dims(global_features, axis=1), axis=1)
        return tf.tensordot(local_features + expanded_g, self.u, axes=1)

    def compute_output_shape(self, input_shape):
        local_feature_shape, global_feature_shape = input_shape
        return local_feature_shape[0:3] + (1,)
