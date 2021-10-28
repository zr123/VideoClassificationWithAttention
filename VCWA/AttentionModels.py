import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.applications import resnet
from VCWA import ResidualAttentionModule, CBAM
from VCWA.AttentionModule import AttentionModule
from VCWA.AttentionGate import AttentionGate


HEIGHT = 224
WIDTH = 224
CHANNELS = 3
CLASSES = 51


def create_L2PA_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, weights=None):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, include_top=False, weights=weights)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = layers.GlobalAveragePooling2D()(basenet.output)

    attention1 = AttentionModule()([local1.output, global_features])
    attention2 = AttentionModule()([local2.output, global_features])
    attention3 = AttentionModule()([local3.output, global_features])

    model_output = tf.keras.layers.Concatenate()([
        layers.Flatten()(attention1),
        layers.Flatten()(attention2),
        layers.Flatten()(attention3)
    ])
    model_output = tf.keras.layers.Dense(1024, activation="relu")(model_output)
    model_output = layers.Dropout(0.5)(model_output)
    model_output = tf.keras.layers.Dense(classes, activation="softmax")(model_output)

    train_model = Model(inputs=input_layer, outputs=model_output, name="L2PA_ResNet50v2")
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


def create_AttentionGated_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, weights=None):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, include_top=False, weights=weights)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = layers.GlobalAveragePooling2D()(basenet.output)

    a1 = AttentionGate(64)([local1.output, global_features])
    a2 = AttentionGate(64)([local2.output, global_features])
    a3 = AttentionGate(64)([local3.output, global_features])

    attended_features1 = layers.Multiply()([local1.output, a1])
    attended_features1 = layers.GlobalAveragePooling2D()(attended_features1)
    attended_features1 = layers.Dense(2048, activation="relu")(attended_features1)
    attended_features1 = layers.Dense(classes, activation="softmax")(attended_features1)

    attended_features2 = layers.Multiply()([local2.output, a2])
    attended_features2 = layers.GlobalAveragePooling2D()(attended_features2)
    attended_features2 = layers.Dense(2048, activation="relu")(attended_features2)
    attended_features2 = layers.Dense(classes, activation="softmax")(attended_features2)

    attended_features3 = layers.Multiply()([local3.output, a3])
    attended_features3 = layers.GlobalAveragePooling2D()(attended_features3)
    attended_features3 = layers.Dense(2048, activation="relu")(attended_features3)
    attended_features3 = layers.Dense(classes, activation="softmax")(attended_features3)

    top = layers.Dense(classes, activation="softmax")(global_features)

    final = layers.Average()([attended_features1, attended_features2, attended_features3, top])

    model = tf.keras.models.Model(inputs=input_layer, outputs=final, name="AttGated_ResNet50v2")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_AttentionGatedGrid_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, weights=None):
    basenet = tf.keras.applications.ResNet50V2(input_shape=input_shape, classes=classes, include_top=False, weights=weights)
    input_layer = basenet.input

    for layer in basenet.layers:
        if layer.name == "conv2_block3_out":
            local1 = layer
        if layer.name == "conv3_block4_out":
            local2 = layer
        if layer.name == "conv4_block6_out":
            local3 = layer

    global_features = basenet.output

    a1 = AttentionGate(64, grid_attention=True)([local1.output, global_features])
    a2 = AttentionGate(64, grid_attention=True)([local2.output, global_features])
    a3 = AttentionGate(64, grid_attention=True)([local3.output, global_features])

    attended_features1 = layers.Multiply()([local1.output, a1])
    attended_features1 = tf.keras.layers.GlobalAveragePooling2D()(attended_features1)
    attended_features1 = layers.Dense(2048, activation="relu")(attended_features1)
    attended_features1 = layers.Dense(classes, activation="softmax")(attended_features1)

    attended_features2 = layers.Multiply()([local2.output, a2])
    attended_features2 = tf.keras.layers.GlobalAveragePooling2D()(attended_features2)
    attended_features2 = layers.Dense(2048, activation="relu")(attended_features2)
    attended_features2 = layers.Dense(classes, activation="softmax")(attended_features2)

    attended_features3 = layers.Multiply()([local3.output, a3])
    attended_features3 = tf.keras.layers.GlobalAveragePooling2D()(attended_features3)
    attended_features3 = layers.Dense(2048, activation="relu")(attended_features3)
    attended_features3 = layers.Dense(classes, activation="softmax")(attended_features3)

    top = layers.GlobalAveragePooling2D()(basenet.output)
    top = layers.Dense(classes, activation="softmax")(top)

    final = layers.Average()([attended_features1, attended_features2, attended_features3, top])

    model = tf.keras.models.Model(inputs=input_layer, outputs=final, name="AttGatedGrid_ResNet50v2")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_ResidualAttention_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, weights=None):
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

    basenet = resnet.ResNet(
        stack_fn,
        True,
        True,
        'ResAttentionNet50v2',
        include_top=False,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling='avg',
        classifier_activation="softmax")

    top = layers.Dense(classes)(basenet.output)
    model = Model(inputs=basenet.inputs, outputs=top, name="ResAttentionNet50v2")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def create_CBAM_ResNet50v2(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, weights=None):
    def CBAM_stack(x, filters, blocks, stride1=2, name=None):
        x = resnet.block2(x, filters, conv_shortcut=True, name=name + '_block1')
        for i in range(2, blocks):
            x = resnet.block2(x, filters, name=name + '_block' + str(i))

        # CBAM module inserted here
        f_dashdash = CBAM.create_residual_attention_module(x)
        layers.Add()([x, f_dashdash])

        x = resnet.block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
        return x

    def stack_fn(x):
        x = CBAM_stack(x, 64, 3, name='conv2')
        x = CBAM_stack(x, 128, 4, name='conv3')
        x = CBAM_stack(x, 256, 6, name='conv4')
        return resnet.stack2(x, 512, 3, stride1=1, name='conv5')

    basenet = resnet.ResNet(
        stack_fn,
        True,
        True,
        'CBAM_Resnet50v2',
        include_top=False,
        weights=weights,
        input_tensor=None,
        input_shape=input_shape,
        pooling='avg',
        classifier_activation="softmax")

    top = layers.Dense(classes)(basenet.output)
    model = Model(inputs=basenet.inputs, outputs=top, name="CBAM_Resnet50v2")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def tiny_cnn(input_shape=(HEIGHT, WIDTH, CHANNELS), classes=CLASSES, additional_batchnorm=False):
    model = Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Conv2D(96, 7, 2, activation="relu"))
    model.add(layers.MaxPooling2D(3, 2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, 5, 2, activation="relu"))
    model.add(layers.MaxPooling2D(3, 2))
    if additional_batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, 3, 1, activation="relu"))
    model.add(layers.Conv2D(512, 3, 1, activation="relu"))
    model.add(layers.Conv2D(512, 3, 1, activation="relu"))
    model.add(layers.MaxPooling2D(3, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu", name="spatial_full6"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation="relu", name="spatial_full7"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation="softmax", name="spatial_softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
