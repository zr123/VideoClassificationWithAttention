import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ReLU, Conv1D, Dense, Flatten, Reshape
from tensorflow.keras.activations import softmax, sigmoid
import AttentionCommon


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
            attention = AttentionCommon.softmax2d(compatibility, name=None) # softmax(compatibility, axis=-1)
        if self.attention_function == "sigmoid":
            attention = AttentionCommon.sigmoid2d(compatibility, name=None) # sigmoid(compatibility, axis=-1)
        if self.attention_function == "pseudo-softmax":
            attention = AttentionCommon.pseudo_softmax2d(compatibility, name=None) # self.pseudo_softmax(compatibility)

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
        if self.grid_attention:
            row = int(local_features.shape[1] / global_features.shape[1])
            col = int(local_features.shape[2] / global_features.shape[2])
            global_features = tf.keras.layers.UpSampling2D((row, col), interpolation='bilinear')(global_features)
        global_component = self.global_transformation(global_features)
        # expand g from (batch_size, inter_channels) to (batch_size, local_width, local_height, inter_channels)
        if not self.grid_attention:
            global_component = tf.expand_dims(tf.expand_dims(global_component, axis=1), axis=1)
        compatibility = self.joining_Conv1D(ReLU()(local_component + global_component))
        return compatibility
