import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.applications import InceptionResNetV2

# downscaled defaults for hmdb51
HEIGHT = 224
WIDTH = 224
FRAMES = 40
CHANNELS = 3
CLASSES = 51


def create_2DCNN_MLP(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="2DCNN_MLP")
    model.add(Input(input_shape))
    # 2D conv layers with time distribution
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    # dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    # finalize
    model.add(layers.Dense(classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_3DCNN(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="3DCNN")
    model.add(Input(input_shape))
    # 3D conv layers
    model.add(layers.Conv3D(32, 3, activation="relu"))
    model.add(layers.MaxPooling3D())
    model.add(layers.Conv3D(32, 3, activation="relu"))
    model.add(layers.MaxPooling3D())
    model.add(layers.Conv3D(32, 3, activation="relu"))
    model.add(layers.MaxPooling3D())
    # dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    # finalize
    model.add(layers.Dense(classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_LSTM(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="LSTM")
    model.add(Input(input_shape))
    # 2D conv Layers with time distribution
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    # LSTM
    model.add(layers.LSTM(1024))
    # dense layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    # finalize
    model.add(layers.Dense(classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def assemble_lstm(basenet, classes, recreate_top=False):
    inputs = layers.Input(basenet.inputs[0].shape)
    if recreate_top:
        x = layers.TimeDistributed(recreate_top_fn(basenet, classes))(inputs)
    else:
        x = layers.TimeDistributed(basenet)(inputs)

    x = layers.LSTM(1024, return_sequences=False, dropout=0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_Transfer_LSTM(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="TransferLSTM")
    model.add(Input(input_shape))
    model.add(
        layers.TimeDistributed(InceptionResNetV2(
            input_shape=input_shape[1:4],
            include_top=False)))
    # Make transferlearning basemodel weights nontrainable
    model.layers[0].trainable = False
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
    # LSTM
    model.add(layers.LSTM(128))
    # dense layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    # finalize
    model.add(layers.Dense(classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def lstm_test(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="TrailTransferLSTM")
    model.add(Input(input_shape))
    model.add(
        layers.TimeDistributed(InceptionResNetV2(
            input_shape=input_shape[1:4],
            include_top=False)))
    # Make transferlearning basemodel weights nontrainable
    model.layers[0].trainable = False
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))

    # Model.
    model.add(layers.LSTM(1024, return_sequences=False,
                   dropout=0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def assemble_TwoStreamModel(spatial_stream_model, temporal_stream_model, classes, fusion="average", recreate_top=False):
    spatial_stream_input = layers.Input(spatial_stream_model.inputs[0].shape)
    temporal_stream_input = Input(temporal_stream_model.inputs[0].shape)

    if recreate_top:
        spatial_stream = layers.TimeDistributed(recreate_top_fn(spatial_stream_model, classes))(spatial_stream_input)
        temporal_stream = layers.TimeDistributed(recreate_top_fn(temporal_stream_model, classes))(temporal_stream_input)
    else:
        spatial_stream = layers.TimeDistributed(spatial_stream_model)(spatial_stream_input)
        temporal_stream = layers.TimeDistributed(temporal_stream_model)(temporal_stream_input)

    # late fusion
    if fusion == "average":
        fusion = layers.Concatenate(axis=1)([spatial_stream, temporal_stream])
        fusion = tf.math.reduce_mean(fusion, axis=1)

    model = Model(inputs=[spatial_stream_input, temporal_stream_input], outputs=fusion)
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def recreate_top_fn(model, classes):
    inputs = model.inputs
    outputs = layers.Dense(classes)(model.layers[-2].output)
    return Model(inputs=inputs, outputs=outputs)
