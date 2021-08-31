import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Flatten

from AttentionModule import AttentionModule
from AttentionGate import AttentionGate
from tensorflow.python.keras.models import Model

HEIGHT = 128
WIDTH = 128
CHANNELS = 3
CLASSES = 51


def create_L2PA_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=num_classes, weights=None)

    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = basenet.layers[-2]

    attention1 = AttentionModule()([local1.output, global_features.output])
    attention2 = AttentionModule()([local2.output, global_features.output])
    attention3 = AttentionModule()([local3.output, global_features.output])

    model_output = tf.keras.layers.Concatenate()([
        Flatten()(attention1),
        Flatten()(attention2),
        Flatten()(attention3)
    ])
    model_output = tf.keras.layers.Dense(1024, activation="relu")(model_output)
    model_output = Dropout(0.5)(model_output)
    model_output = tf.keras.layers.Dense(num_classes, activation="softmax")(model_output)

    train_model = Model(inputs=input_layer, outputs=model_output)
    train_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return train_model

# get the helper-model to extract the attention information
def get_attention_extractor(model):
    attention = []
    for layer in model.layers:
        if "AttentionModule" in str(type(layer)):
            attention.append(layer.output)
        if "AttentionGate" in str(type(layer)):
            attention.append(layer.output)
    attention_extractor_model = Model(inputs=model.input, outputs=[model.output] + attention)
    return attention_extractor_model


def create_AttentionGated_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
    basenet = tf.keras.applications.ResNet50V2(classes=10, weights=None)

    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = basenet.layers[-2]

    a1 = AttentionGate(32)([local1.output, global_features.output])
    a2 = AttentionGate(32)([local2.output, global_features.output])
    a3 = AttentionGate(32)([local3.output, global_features.output])

    a1 = tf.keras.layers.GlobalAveragePooling2D()(a1)
    a1 = Dense(1024, activation="relu")(a1)
    a1 = Dense(10, activation="softmax")(a1)

    a2 = tf.keras.layers.GlobalAveragePooling2D()(a2)
    a2 = Dense(1024, activation="relu")(a2)
    a2 = Dense(10, activation="softmax")(a2)

    a3 = tf.keras.layers.GlobalAveragePooling2D()(a3)
    a3 = Dense(1024, activation="relu")(a3)
    a3 = Dense(10, activation="softmax")(a3)

    final = (a1 + a2 + a3 + basenet.output) / 4.0

    model = tf.keras.models.Model(inputs=input_layer, outputs=final)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
