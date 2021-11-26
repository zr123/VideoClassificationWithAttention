import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import ReLU, Conv1D, Dense

from VCWA.Attention.shared import softmax2d, pseudo_softmax2d


class AttentionGate(keras.layers.Layer):

    def __init__(self, inter_channels=1, attention_function="pseudo-softmax", grid_attention=False):
        super(AttentionGate, self).__init__()
        assert attention_function in ["softmax", "sigmoid", "pseudo-softmax"], "Unexpected attention_function argument."
        self.inter_channels = inter_channels
        self.attention_function = attention_function
        self.grid_attention = grid_attention

    def call(self, inputs, **kwargs):
        local_features, global_features = inputs
        compatibility = self.compatibility_function(local_features, global_features)

        if self.attention_function == "softmax":
            attention = softmax2d(compatibility, name=None)
        if self.attention_function == "sigmoid":
            attention = sigmoid(compatibility)
        if self.attention_function == "pseudo-softmax":
            attention = pseudo_softmax2d(compatibility, name=None)

        attention = tf.math.reduce_mean(attention, axis=-1, keepdims=True)
        return attention

    def build(self, input_shape):
        local_feature_shape, global_feature_shape = input_shape
        self.local_Conv1D = Conv1D(self.inter_channels, 1, use_bias=False)
        if self.grid_attention:
            self.global_transformation = Conv1D(self.inter_channels, 1)
        else:
            self.global_transformation = Dense(self.inter_channels)
        self.joining_Conv1D = Conv1D(self.inter_channels, 1)

    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({
            'inter_channels':
                self.inter_channels,
            'attention_function':
                self.attention_function
        })
        return config

    def compatibility_function(self, local_features, global_features):
        local_component = self.local_Conv1D(local_features)
        global_component = self.global_transformation(global_features)

        if self.grid_attention:
            row = local_features.shape[1] // global_features.shape[1]
            col = local_features.shape[2] // global_features.shape[2]
            global_component = tf.keras.layers.UpSampling2D((row, col), interpolation='nearest')(global_component)
        else:
            # expand g from (batch_size, inter_channels) to (batch_size, local_width, local_height, inter_channels)
            global_component = tf.expand_dims(tf.expand_dims(global_component, axis=1), axis=1)

        compatibility = self.joining_Conv1D(ReLU()(local_component + global_component))
        return compatibility

    def compute_output_shape(self, input_shape):
        local_feature_shape, global_feature_shape = input_shape
        return local_feature_shape[0:3] + (1,)
