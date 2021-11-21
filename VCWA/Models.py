import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential, Model
from VCWA import AttentionModels, Common
import imageio
import numpy as np
import datetime

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


def assemble_lstm(backbone, classes):
    inputs = layers.Input(backbone.inputs[0].shape)
    # strip the prediction layer and use the cnn-frames as lstm-inputs
    x = layers.TimeDistributed(
        Model(inputs=backbone.inputs, outputs=[backbone.layers[-2].output])
    )(inputs)

    x = layers.LSTM(512, return_sequences=True, dropout=0.5)(x)
    x = layers.LSTM(512, return_sequences=True, dropout=0.5)(x)
    x = layers.LSTM(512, return_sequences=False, dropout=0.5)(x)
    x = layers.Dense(classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=x, name="LSTM_" + backbone.name)


def assemble_TwoStreamModel(spatial_stream_model, temporal_stream_model, classes, fusion="average", recreate_top=False):
    spatial_stream_input = layers.Input(spatial_stream_model.inputs[0].shape)
    temporal_stream_input = layers.Input(temporal_stream_model.inputs[0].shape)

    if recreate_top:
        spatial_stream = layers.TimeDistributed(recreate_top_fn(spatial_stream_model, classes))(spatial_stream_input)
        temporal_stream = layers.TimeDistributed(recreate_top_fn(temporal_stream_model, classes))(temporal_stream_input)
    else:
        spatial_stream = layers.TimeDistributed(spatial_stream_model)(spatial_stream_input)
        temporal_stream = layers.TimeDistributed(temporal_stream_model)(temporal_stream_input)

    # late fusion TODO implement dense fusion and SVM fusion
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


def get_twostream_attention(inputs, model, include_input=True, cmap='inferno'):
    input1, input2, td1, td2 = model.layers[0:4]
    layer_extractor_model = AttentionModels.get_attention_extractor(td1.layer)

    prediction = layer_extractor_model.predict(inputs)
    images = []
    for i in range(inputs.shape[0]):
        attention = []
        # prediction[0] is the class-score output of the network
        for a in prediction[1:]:
            attention.append(a[i])

        overlay = Common.combine_attention(attention)
        combined_image = Common.overlay_attention(inputs[i], overlay, cmap=cmap)

        if include_input:
            combined_image = np.concatenate((inputs[i] / 2 + 0.5, combined_image), axis=1)

        images.append(combined_image)
    return np.array(images)


def get_twostream_gradcam(inputs, model, layer_name, include_input=True, cmap='inferno'):
    input1, input2, td1, td2 = model.layers[0:4]
    attention = Common.get_gradcam_attention(inputs, td1.layer, layer_name=layer_name)

    images = []
    for i in range(inputs.shape[0]):
        overlay = Common.combine_attention([attention[i]])
        combined_image = Common.overlay_attention(inputs[i], overlay, cmap=cmap)

        if include_input:
            combined_image = np.concatenate((inputs[i] / 2 + 0.5, combined_image), axis=1)

        images.append(combined_image)
    return images


def video_to_gif(images, path="./attention.gif", fps=7):
    images = (np.array(images) * 255).astype(np.uint8)
    imageio.mimsave(path, images, fps=fps)


def train_optflow_model(video_model,
           optflow_model,
           video_train_gen,
           video_test_gen,
           optflow_train_gen,
           optflow_test_gen,
           twostream_test_gen,
           iterations=1,
           classes=CLASSES,
           log_basedir="logs/fit_twostream_25_L10/resnet/",
           model_basedir="models/twostream_25_L10/"):
    vid_tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_basedir + "video/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1)
    optflow_tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_basedir + "optflow/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1)
    twostream_tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_basedir + "twostream/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1)

    for i in range(iterations):
        # Video Model
        video_model.fit(
            video_train_gen,
            epochs=1,
            validation_data=video_test_gen,
            callbacks=[vid_tensorboard_callback])

        # Optflow Model
        optflow_model.fit(
            optflow_train_gen,
            epochs=i,
            initial_epoch=i-1,
            validation_data=optflow_test_gen,
            callbacks=[optflow_tensorboard_callback])

        twostream = assemble_TwoStreamModel(video_model, optflow_model, classes, fusion="average", recreate_top=True)
        twostream.evaluate(twostream_test_gen, callbacks=[twostream_tensorboard_callback])

        # save models
        video_model.save(model_basedir + "video/" + video_model.name)
        optflow_model.save(model_basedir + "optflow/" + optflow_model.name)
