import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, \
    LSTM, GlobalAveragePooling2D, BatchNormalization, Dropout, Average
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.applications import InceptionResNetV2

# downscaled defaults for hmdb51
HEIGHT = 128
WIDTH = 128
FRAMES = 40
CHANNELS = 1
CLASSES = 51


def create_2DCNN_MLP(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
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
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_3DCNN(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
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
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_LSTM(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
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
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def create_Transfer_LSTM(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
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
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


def lstm_test(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
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
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model


# typical input_shape=(None, 224, 224, 3)
def create_TwoStreamModel(input_shape=(FRAMES, HEIGHT, WIDTH, CHANNELS), num_classes=CLASSES):
    # Spatial Stream ConvNet
    spatial_stream_input = Input(input_shape)
    spatial_stream = Sequential([
        Conv2D(96, 7, 2, activation="relu", name="spatial_conv1"),
        MaxPooling2D(3, 2),
        BatchNormalization(),
        Conv2D(256, 5, 2, activation="relu", name="spatial_conv2"),
        MaxPooling2D(3, 2),
        BatchNormalization(),
        Conv2D(512, 3, 1, activation="relu", name="spatial_conv3"),
        Conv2D(512, 3, 1, activation="relu", name="spatial_conv4"),
        Conv2D(512, 3, 1, activation="relu", name="spatial_conv5"),
        MaxPooling2D(3, 2),
        Flatten(),
        Dense(4096, activation="relu", name="spatial_full6"),
        Dropout(0.5),
        Dense(2048, activation="relu", name="spatial_full7"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax", name="spatial_softmax")
    ])
    spatial_stream = TimeDistributed(spatial_stream)(spatial_stream_input)

    # Temporal Stream ConvNet
    temporal_stream_input = Input(input_shape)
    temporal_stream = Sequential([
        Conv2D(96, 7, 2, activation="relu", name="temporal_conv1"),
        MaxPooling2D(3, 2),
        BatchNormalization(),
        Conv2D(256, 5, 2, activation="relu", name="temporal_conv2"),
        MaxPooling2D(3, 2),
        BatchNormalization(),
        Conv2D(512, 3, 1, activation="relu", name="temporal_conv3"),
        Conv2D(512, 3, 1, activation="relu", name="temporal_conv4"),
        Conv2D(512, 3, 1, activation="relu", name="temporal_conv5"),
        MaxPooling2D(3, 2),
        Flatten(),
        Dense(4096, activation="relu", name="temporal_full6"),
        Dropout(0.5),
        Dense(2048, activation="relu", name="temporal_full7"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax", name="temporal_softmax")
    ])
    temporal_stream = TimeDistributed(temporal_stream)(temporal_stream_input)

    # late fusion
    fusion = Average()([spatial_stream, temporal_stream])
    fusion = tf.math.reduce_mean(fusion, axis=1)

    model = Model(inputs=[spatial_stream_input, temporal_stream_input], outputs=fusion)
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.TopKCategoricalAccuracy(5)])
    return model
