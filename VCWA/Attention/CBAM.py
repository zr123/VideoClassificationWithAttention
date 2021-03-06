import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential


def create_cbam_module(x, r=16.0, k=7, **kwargs):
    """CBAM: Convolutional Block Attention Module by Woo et al.

        Arguments:
            x: input tensor
            r: reduction ratio of the hidden layer in the channel attention block
            k: kernel-size of the Conv2D layer in the spatial attention block
    """
    f_dash = create_channel_attention_block(x, r)
    f_dashdash = create_spatial_attention_block(f_dash, k)
    x = layers.Add()([x, f_dashdash])
    return x


def create_channel_attention_block(x, r):
    channel_count = x.shape[3]
    f_c_avg = layers.GlobalAveragePooling2D()(x)
    f_c_max = layers.GlobalMaxPooling2D()(x)
    # shared mlp
    mlp = Sequential([
        layers.Dense(channel_count / r, activation="relu"),
        layers.Dense(channel_count)
    ])
    channel_attention = layers.Activation(activations.sigmoid)(
        layers.Add()([mlp(f_c_avg), mlp(f_c_max)])
    )
    # expand tensor shape to (batch_size, 1, 1, channel_count) so we can properly multiply with x
    channel_attention = layers.Lambda(lambda z: tf.expand_dims(tf.expand_dims(z, axis=1), axis=1))(channel_attention)
    channel_attention = layers.Multiply()([x, channel_attention])
    return channel_attention


def create_spatial_attention_block(x, k):
    f_s_avg = layers.Lambda(lambda z: tf.math.reduce_mean(z, axis=-1, keepdims=True))(x)  # "channel-wise avg-pooling"
    f_s_max = layers.Lambda(lambda z: tf.math.reduce_max(z, axis=-1, keepdims=True))(x)  # "channel-wise max-pooling"
    combined_spatial_features = layers.Concatenate()([f_s_avg, f_s_max])
    spatial_attention = layers.Conv2D(1, k, padding="same", activation="sigmoid")(combined_spatial_features)
    spatial_attention = layers.Multiply()([x, spatial_attention])
    return spatial_attention
