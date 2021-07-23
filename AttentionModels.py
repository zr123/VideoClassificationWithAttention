import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Flatten

from AttentionModule import AttentionModule
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
    model_output = tf.keras.layers.Dense(10, activation="softmax")(model_output)

    train_model = Model(inputs=input_layer, outputs=model_output)
    train_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    attention_extractor_model = Model(inputs=input_layer, outputs=[model_output, attention1, attention2, attention3])

    return train_model, attention_extractor_model
