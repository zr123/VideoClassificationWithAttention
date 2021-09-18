import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, \
    LSTM, GlobalAveragePooling2D, BatchNormalization, Dropout, Average
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
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    # dense layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    # finalize
    model.add(Dense(classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_3DCNN(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="3DCNN")
    model.add(Input(input_shape))
    # 3D conv layers
    model.add(Conv3D(32, 3, activation="relu"))
    model.add(MaxPooling3D())
    model.add(Conv3D(32, 3, activation="relu"))
    model.add(MaxPooling3D())
    model.add(Conv3D(32, 3, activation="relu"))
    model.add(MaxPooling3D())
    # dense layers
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    # finalize
    model.add(Dense(classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_LSTM(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="LSTM")
    model.add(Input(input_shape))
    # 2D conv Layers with time distribution
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    # LSTM
    model.add(LSTM(1024))
    # dense layers
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    # finalize
    model.add(Dense(classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_Transfer_LSTM(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="TransferLSTM")
    model.add(Input(input_shape))
    model.add(
        TimeDistributed(InceptionResNetV2(
            input_shape=input_shape[1:4],
            include_top=False)))
    # Make transferlearning basemodel weights nontrainable
    model.layers[0].trainable = False
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    # LSTM
    model.add(LSTM(128))
    # dense layers
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    # finalize
    model.add(Dense(classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def lstm_test(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), classes=CLASSES):
    model = Sequential(name="TrailTransferLSTM")
    model.add(Input(input_shape))
    model.add(
        TimeDistributed(InceptionResNetV2(
            input_shape=input_shape[1:4],
            include_top=False)))
    # Make transferlearning basemodel weights nontrainable
    model.layers[0].trainable = False
    model.add(TimeDistributed(GlobalAveragePooling2D()))

    # Model.
    model.add(LSTM(1024, return_sequences=False,
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model



def wrapper_ResNet(input_shape=None, classes=1000, fn=tf.keras.applications.ResNet50V2):
    """ Small compatability wrapper for additional arguments like weights=None"""
    return fn(input_shape=input_shape, classes=classes, weights=None)


def create_TwoStreamModel(
        input_shape=(None, HEIGHT, WIDTH, 3),
        optflow_shape=(None, HEIGHT, WIDTH, 20),
        classes=CLASSES,
        fn_create_base_model=wrapper_ResNet,
        fusion="average"):
    assert fusion in ["average"], "Unknown parameter for fusion: " + str(fusion)

    # Spatial Stream 2D-ConvNet
    spatial_stream_input = Input(input_shape)
    spatial_stream = fn_create_base_model(input_shape=input_shape[1:4], classes=classes)
    spatial_stream = TimeDistributed(spatial_stream)(spatial_stream_input)

    # Temporal Stream 2D-ConvNet
    temporal_stream_input = Input(optflow_shape)
    temporal_stream = fn_create_base_model(input_shape=optflow_shape[1:4], classes=classes)
    temporal_stream = TimeDistributed(temporal_stream)(temporal_stream_input)

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
