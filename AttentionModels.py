import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras import layers
from AttentionModule import AttentionModule
from AttentionGate import AttentionGate
from tensorflow.python.keras.models import Model
import ResidualAttentionModule
from tensorflow.python.keras.applications import resnet
import CBAM

HEIGHT = 128
WIDTH = 128
CHANNELS = 3
CLASSES = 51


def create_L2PA_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, weights=None)

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
    model_output = tf.keras.layers.Dense(classes, activation="softmax")(model_output)

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
        if "_ResidualAttention" in layer.name:
            attention.append(tf.math.reduce_mean(layer.output, axis=-1))

    attention_extractor_model = Model(inputs=model.input, outputs=[model.output] + attention)
    return attention_extractor_model


def create_AttentionGated_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, weights=None)

    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = basenet.layers[-2]

    a1 = AttentionGate(64)([local1.output, global_features.output])
    a2 = AttentionGate(64)([local2.output, global_features.output])
    a3 = AttentionGate(64)([local3.output, global_features.output])


    attended_features1 = layers.Multiply()([local1.output, a1])
    attended_features1 = tf.keras.layers.GlobalAveragePooling2D()(attended_features1)
    attended_features1 = Dense(2048, activation="relu")(attended_features1)
    attended_features1 = Dense(classes, activation="softmax")(attended_features1)

    attended_features2 = layers.Multiply()([local2.output, a2])
    attended_features2 = tf.keras.layers.GlobalAveragePooling2D()(attended_features2)
    attended_features2 = Dense(2048, activation="relu")(attended_features2)
    attended_features2 = Dense(classes, activation="softmax")(attended_features2)

    attended_features3 = layers.Multiply()([local3.output, a3])
    attended_features3 = tf.keras.layers.GlobalAveragePooling2D()(attended_features3)
    attended_features3 = Dense(2048, activation="relu")(attended_features3)
    attended_features3 = Dense(classes, activation="softmax")(attended_features3)

    final = layers.Average()([attended_features1, attended_features2, attended_features3, basenet.output])

    model = tf.keras.models.Model(inputs=input_layer, outputs=final)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_AttentionGatedGrid_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, weights=None)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = basenet.layers[-3]

    a1 = AttentionGate(64, grid_attention=True)([local1.output, global_features.output])
    a2 = AttentionGate(64, grid_attention=True)([local2.output, global_features.output])
    a3 = AttentionGate(64, grid_attention=True)([local3.output, global_features.output])

    attended_features1 = layers.Multiply()([local1.output, a1])
    attended_features1 = tf.keras.layers.GlobalAveragePooling2D()(attended_features1)
    attended_features1 = Dense(2048, activation="relu")(attended_features1)
    attended_features1 = Dense(classes, activation="softmax")(attended_features1)

    attended_features2 = layers.Multiply()([local2.output, a2])
    attended_features2 = tf.keras.layers.GlobalAveragePooling2D()(attended_features2)
    attended_features2 = Dense(2048, activation="relu")(attended_features2)
    attended_features2 = Dense(classes, activation="softmax")(attended_features2)

    attended_features3 = layers.Multiply()([local3.output, a3])
    attended_features3 = tf.keras.layers.GlobalAveragePooling2D()(attended_features3)
    attended_features3 = Dense(2048, activation="relu")(attended_features3)
    attended_features3 = Dense(classes, activation="softmax")(attended_features3)

    final = layers.Average()([attended_features1, attended_features2, attended_features3, basenet.output])

    model = tf.keras.models.Model(inputs=input_layer, outputs=final)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_ResidualAttention_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    def ResidualAttention_stack(x, filters, blocks, shortcuts, stride1=2, name=None):
        x = resnet.block2(x, filters, conv_shortcut=True, name=name + '_block1')
        for i in range(2, blocks):
            x = resnet.block2(x, filters, name=name + '_block' + str(i))
        # residual attention module inserted here
        x = ResidualAttentionModule.create_residual_attention_module(x, filters, shortcuts=shortcuts, name=name + "_attn", attention_function="pseudo-softmax")
        x = resnet.block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
        return x

    def stack_fn(x):
        x = ResidualAttention_stack(x, 64, 3, shortcuts=2, name='conv2')
        x = ResidualAttention_stack(x, 128, 4, shortcuts=1, name='conv3')
        x = ResidualAttention_stack(x, 256, 6, shortcuts=0, name='conv4')
        return resnet.stack2(x, 512, 3, stride1=1, name='conv5')

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        'attention_resnet50v2',
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=classes,
        classifier_activation="softmax")


def create_CBAM_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    def CBAM_stack(x, filters, blocks, stride1=2, name=None):
        x = resnet.block2(x, filters, conv_shortcut=True, name=name + '_block1')
        for i in range(2, blocks):
            x = resnet.block2(x, filters, name=name + '_block' + str(i))

        # CBAM module inserted here
        f_dashdash = CBAM.create_residual_attention_module(x)
        x = x + f_dashdash

        x = resnet.block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
        return x

    def stack_fn(x):
        x = CBAM_stack(x, 64, 3, name='conv2')
        x = CBAM_stack(x, 128, 4, name='conv3')
        x = CBAM_stack(x, 256, 6, name='conv4')
        return resnet.stack2(x, 512, 3, stride1=1, name='conv5')

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        'attention_resnet50v2',
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=classes,
        classifier_activation="softmax")