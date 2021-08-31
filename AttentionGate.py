import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import ReLU, Conv1D, Dense, Flatten, Reshape
from tensorflow.keras.activations import softmax, sigmoid



class AttentionGate(keras.layers.Layer):

    def __init__(self, inter_channels=1, attention_function="pseudo-softmax"):
        super(AttentionGate, self).__init__()
        assert attention_function in ["softmax", "sigmoid", "pseudo-softmax"], "Unexpected attention_function argument."
        self.inter_channels = inter_channels
        self.attention_function = attention_function

    def call(self, inputs, **kwargs):
        local_features, global_features = inputs
        compatibility = self.compatibility_function(local_features, global_features)

        if self.attention_function == "softmax":
            attention = softmax(compatibility, axis=-1)
        if self.attention_function == "sigmoid":
            attention = sigmoid(compatibility, axis=-1)
        if self.attention_function == "pseudo-softmax":
            attention = self.pseudo_softmax(compatibility)
        return attention

    # pseudo-softmax: subtract by min value and divide by sum -> less sparse but similar properties as softmax
    def pseudo_softmax(self, compatibility):
        shape = compatibility.shape
        attention = Flatten()(compatibility)
        a_min = tf.math.reduce_min(attention, axis=-1, keepdims=True)
        a_sum = tf.math.reduce_sum(attention, axis=-1, keepdims=True)
        attention = (attention - a_min) / a_sum
        attention = Reshape(shape[1:])(attention)
        return attention

    def build(self, input_shape):
        local_feature_shape, global_feature_shape = input_shape
        self.local_Conv1D = Conv1D(self.inter_channels, 1, use_bias=False)
        self.global_Dense  = Dense(self.inter_channels)
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
        global_component = self.global_Dense(global_features)
        # expand g from (batch_size, inter_channels) to (batch_size, local_width, local_height, inter_channels)
        expanded_g = tf.expand_dims(tf.expand_dims(global_component, axis=1), axis=1)
        compatibility = self.joining_Conv1D(ReLU()(local_component + expanded_g))
        return compatibility
