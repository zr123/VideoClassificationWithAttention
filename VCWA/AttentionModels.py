import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from VCWA import ResidualAttentionModule, CBAM
from VCWA.L2PAModule import L2PAModule
from VCWA.AttentionGate import AttentionGate
from TF_modification import resatt_mobilenet_v2, mobilenet_v2


HEIGHT = 224
WIDTH = 224
CHANNELS = 3
CLASSES = 51


# get the helper-model to extract the attention information
def get_attention_extractor(model):
    attention = []
    for layer in model.layers:
        if "AttentionModule" in str(type(layer)):
            attention.append(layer.output)
        if "AttentionGate" in str(type(layer)):
            attention.append(layer.output)
        if "_ResidualAttention" in layer.name:
            attention.append(tf.math.reduce_mean(layer.output, axis=-1))

    attention_extractor_model = Model(inputs=model.input, outputs=[model.output] + attention)
    return attention_extractor_model


def create_L2PA_MobileNetV2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        classes=CLASSES,
        name="L2PA_MobileNetV2",
        basenet_fn=mobilenet_v2.MobileNetV2,
        layer_names=None):
    if layer_names is None:
        layer_names = ["block_5_add", "block_12_add", "block_15_add"]
    basenet = basenet_fn(
        input_shape=input_shape,
        classes=classes,
        include_top=False,
        weights=None)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == layer_names[0]:
            local1 = layer
        if layer.name == layer_names[1]:
            local2 = layer
        if layer.name == layer_names[2]:
            local3 = layer

    global_features = layers.GlobalAveragePooling2D()(basenet.output)

    attention1 = L2PAModule()([local1.output, global_features])
    attention2 = L2PAModule()([local2.output, global_features])
    attention3 = L2PAModule()([local3.output, global_features])

    model_output = tf.keras.layers.Concatenate()([
        layers.Flatten()(attention1),
        layers.Flatten()(attention2),
        layers.Flatten()(attention3)
    ])
    model_output = tf.keras.layers.Dense(1024, activation="relu")(model_output)
    model_output = layers.Dropout(0.5)(model_output)
    model_output = tf.keras.layers.Dense(classes, activation="softmax")(model_output)

    return Model(inputs=input_layer, outputs=model_output, name=name)


def create_AttentionGated_MobileNetV2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        classes=CLASSES,
        name="AttGated_MobileNetV2",
        basenet_fn=mobilenet_v2.MobileNetV2,
        layer_names=None,
        internal_dimensions=32,
        final_dense_size=1024):
    if layer_names is None:
        layer_names = ["block_5_add", "block_12_add", "block_15_add"]
    basenet = basenet_fn(
        input_shape=input_shape,
        classes=classes,
        include_top=False,
        weights=None)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == layer_names[0]:
            local1 = layer
        if layer.name == layer_names[1]:
            local2 = layer
        if layer.name == layer_names[2]:
            local3 = layer

    global_features = layers.GlobalAveragePooling2D()(basenet.output)

    a1 = AttentionGate(internal_dimensions)([local1.output, global_features])
    a2 = AttentionGate(internal_dimensions)([local2.output, global_features])
    a3 = AttentionGate(internal_dimensions)([local3.output, global_features])

    attended_features1 = layers.Multiply()([local1.output, a1])
    attended_features1 = layers.GlobalAveragePooling2D()(attended_features1)
    attended_features1 = layers.Dense(final_dense_size, activation="relu")(attended_features1)
    attended_features1 = layers.Dense(classes, activation="softmax")(attended_features1)

    attended_features2 = layers.Multiply()([local2.output, a2])
    attended_features2 = layers.GlobalAveragePooling2D()(attended_features2)
    attended_features2 = layers.Dense(final_dense_size, activation="relu")(attended_features2)
    attended_features2 = layers.Dense(classes, activation="softmax")(attended_features2)

    attended_features3 = layers.Multiply()([local3.output, a3])
    attended_features3 = layers.GlobalAveragePooling2D()(attended_features3)
    attended_features3 = layers.Dense(final_dense_size, activation="relu")(attended_features3)
    attended_features3 = layers.Dense(classes, activation="softmax")(attended_features3)

    top = layers.Dense(classes, activation="softmax")(global_features)

    final = layers.Average()([attended_features1, attended_features2, attended_features3, top])

    return tf.keras.models.Model(inputs=input_layer, outputs=final, name=name)


def create_AttentionGatedGrid_MobileNetV2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        classes=CLASSES,
        name="AttGatedGrid_MobileNetV2",
        basenet_fn=mobilenet_v2.MobileNetV2,
        layer_names=None,
        internal_dimensions=32,
        final_dense_size=1024):
    if layer_names is None:
        layer_names = ["block_5_add", "block_12_add", "block_15_add"]
    basenet = basenet_fn(
        input_shape=input_shape,
        classes=classes,
        include_top=False,
        weights=None)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == layer_names[0]:
            local1 = layer
        if layer.name == layer_names[1]:
            local2 = layer
        if layer.name == layer_names[2]:
            local3 = layer

    global_features = basenet.output

    a1 = AttentionGate(internal_dimensions, grid_attention=True)([local1.output, global_features])
    a2 = AttentionGate(internal_dimensions, grid_attention=True)([local2.output, global_features])
    a3 = AttentionGate(internal_dimensions, grid_attention=True)([local3.output, global_features])

    attended_features1 = layers.Multiply()([local1.output, a1])
    attended_features1 = tf.keras.layers.GlobalAveragePooling2D()(attended_features1)
    attended_features1 = layers.Dense(final_dense_size, activation="relu")(attended_features1)
    attended_features1 = layers.Dense(classes, activation="softmax")(attended_features1)

    attended_features2 = layers.Multiply()([local2.output, a2])
    attended_features2 = tf.keras.layers.GlobalAveragePooling2D()(attended_features2)
    attended_features2 = layers.Dense(final_dense_size, activation="relu")(attended_features2)
    attended_features2 = layers.Dense(classes, activation="softmax")(attended_features2)

    attended_features3 = layers.Multiply()([local3.output, a3])
    attended_features3 = tf.keras.layers.GlobalAveragePooling2D()(attended_features3)
    attended_features3 = layers.Dense(final_dense_size, activation="relu")(attended_features3)
    attended_features3 = layers.Dense(classes, activation="softmax")(attended_features3)

    top = layers.GlobalAveragePooling2D()(basenet.output)
    top = layers.Dense(classes, activation="softmax")(top)

    final = layers.Average()([attended_features1, attended_features2, attended_features3, top])

    return tf.keras.models.Model(inputs=input_layer, outputs=final, name=name)


def create_ResidualAttention_MobileNetV2(
        input_shape=(HEIGHT, WIDTH, CHANNELS),
        classes=CLASSES,
        name='ResAttentionMobileNetV2'):
    model = resatt_mobilenet_v2.MobileNetV2(
        input_shape=input_shape,
        classes=classes,
        weights=None)
    model._name = name
    return model


def create_CBAM_MobileNetV2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, name='CBAM_MobileNetV2'):
    model = mobilenet_v2.MobileNetV2(
        input_shape=input_shape,
        classes=classes,
        attention_builder_fn=CBAM.create_cbam_module,
        weights=None)
    model._name = name
    return model
