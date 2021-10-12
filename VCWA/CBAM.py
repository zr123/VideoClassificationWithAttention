import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential


def create_residual_attention_module(x, r=16.0, k=7):
    """CBAM: Convolutional Block Attention Module by Woo et al.

        Arguments:
            x: input tensor
            r: reduction ratio of the hidden layer in the channel attention block
            k: kernel-size of the Conv2D layer in the spatial attention block
    """
    channels = x.shape[3]
    f_dash = create_channel_attention_block(x, r, channels)
    f_dashdash = create_spatial_attention_block(f_dash, k)
    return f_dashdash


def create_channel_attention_block(x, r, channel_count):
    f_c_avg = layers.GlobalAveragePooling2D()(x)
    f_c_max = layers.GlobalMaxPooling2D()(x)
    mlp = Sequential([
        layers.Dense(channel_count / r, activation="relu"),
        layers.Dense(channel_count)
    ])
    channel_attention = activations.sigmoid(mlp(f_c_avg) + mlp(f_c_max))
    # expand tensor shape to (batch_size, 1, 1, channel_count) so we can properly multiply with x
    channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)
    channel_attention = layers.Multiply()([x, channel_attention])
    return channel_attention


def create_spatial_attention_block(x, k):
    f_s_avg = tf.math.reduce_mean(x, axis=-1, keepdims=True)  # "channel-wise avg-pooling"
    f_s_max = tf.math.reduce_max(x, axis=-1, keepdims=True)  # "channel-wise max-pooling"
    combined_spatial_features = layers.Concatenate()([f_s_avg, f_s_max])
    spatial_attention = layers.Conv2D(1, k, padding="same", activation="sigmoid")(combined_spatial_features)
    spatial_attention = layers.Multiply()([x, spatial_attention])
    return spatial_attention
