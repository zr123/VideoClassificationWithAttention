import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Conv3D, MaxPooling3D, \
    LSTM, GlobalAveragePooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.applications import InceptionResNetV2

# downscaled defaults for hmdb51
HEIGHT = 128
WIDTH = 128
FRAMES = 40
CHANNELS = 1
CLASSES = 51


class Models:

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
