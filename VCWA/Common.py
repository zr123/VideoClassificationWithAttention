import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.python.keras.models import Model
from VCWA import AttentionModels
from lime import lime_image
from skimage.segmentation import mark_boundaries

####################
# Dataset Handling #
####################

def get_dataset(path, split_path, optflow_path=None, split_no=1, dataset_type=None):
    accepted_datasets = ["hmdb51", "ucf101"]
    assert dataset_type in accepted_datasets, "Unexpected dataset " + dataset_type + " not in " + str(accepted_datasets)
    dataset = evaluate_dataset(path)
    if dataset_type == "hmdb51":
        split_df = get_hmdb51_split(split_path, split_no=split_no)
        dataset = dataset.merge(split_df, on="filename")
    if dataset_type == "ucf101":
        split_df = get_ucf101_split(split_path, split_no=split_no)
        dataset = dataset.merge(split_df, on="filename")

    if optflow_path is not None:
        optflow_dataset = evaluate_dataset(optflow_path)
        optflow_dataset["filename"] = optflow_dataset["filename"].str.split(".", expand=True)[0] + ".avi"
        optflow_dataset.rename(columns={"path": "optflow_path"}, inplace=True)
        dataset = dataset.merge(optflow_dataset[["optflow_path", "filename"]], on="filename")

    dataset.category = preprocessing.LabelEncoder().fit_transform(dataset.category)
    return dataset


def evaluate_dataset(path="D:/datasets/hmdb51_org", shuffle=False, random_state=42):
    df = pd.DataFrame()
    for path, directories, files in os.walk(path):
        for f in files:
            path = path.replace("\\", "/")
            df = df.append({
                "path": path + "/" + f,
                "filename": f,
                "category": path.split("/")[-1]},
                ignore_index=True)
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def get_hmdb51_split(path="D:/datasets/hmdb51_org_splits", split_no=1):
    assert split_no in [1, 2, 3], "split_no must be 1, 2, 3, got: " + str(split_no)

    split_df = pd.DataFrame(columns=["filename", "split"])
    for path, directories, files in os.walk(path):
        for file in files:
            if "_split" + str(split_no) + ".txt" in file:
                fo = open(path + "/" + file, "r")
                lines = fo.readlines()
                fo.close()
                for line in lines:
                    filename, split = line.split()
                    split_df.loc[len(split_df)] = (filename, int(split))
    return split_df


def get_ucf101_split(path="D:/datasets/ucfTrainTestlist", split_no=1):
    split_df = pd.DataFrame(columns=["filename", "split"])
    with open(path + "/trainlist0" + str(split_no) + ".txt") as file:
        for line in file.readlines():
            filename, _ = line.rstrip().split()
            _, filename = filename.split("/")
            split_df = split_df.append({"filename": filename, "split": 0}, ignore_index=True)

    with open(path + "/testlist0" + str(split_no) + ".txt") as file:
        for line in file.readlines():
            _, filename = line.rstrip().split("/")
            split_df = split_df.append({"filename": filename, "split": 1}, ignore_index=True)

    return split_df


def load_video(path):
    cap = cv2.VideoCapture(path)
    video = []
    retval, image = cap.read()
    if not retval:
        raise Exception("Invalid path: " + path)
    while retval:
        video.append(image)
        retval, image = cap.read()
    cap.release()
    return np.array(video)


def resize_video(video, resize_shape):
    resized_video = [
        cv2.resize(frame, resize_shape) for frame in video
    ]
    return np.array(resized_video)


def downsample_video(video, downsampling_frames):
    framecount = video.shape[0]
    sampled_frames = np.arange(0, framecount - 1/downsampling_frames, framecount / downsampling_frames)
    downsampled_video = [video[int(f)] for f in sampled_frames]
    return np.array(downsampled_video)


def grayscale_video(video):
    grayscaled_video = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video
    ]
    return np.array(grayscaled_video)


def get_formatted_video(path, resize_shape=None, grayscale=False, downsampling_frames=None, proprocessing_fn=None):
    video = load_video(path)
    if resize_shape is not None:
        video = resize_video(video, resize_shape)
    if downsampling_frames is not None:
        video = downsample_video(video, downsampling_frames)
    if grayscale:
        video = grayscale_video(video)
        video = video.reshape(video.shape[0:3] + (1,))
    if proprocessing_fn is not None:
        video = proprocessing_fn(video)
    return video


def save_video(videoarray, filename):
    frames, height, width = videoarray.shape[0:3]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if len(videoarray.shape) == 4:
        out = cv2.VideoWriter(filename, fourcc, 10, (width, height))
    else:  # video is grayscale
        out = cv2.VideoWriter(filename, fourcc, 10, (width, height), 0)
    for frame in videoarray:
        out.write(frame)
    out.release()


def save_images(videoarray, filename):
    filename, _ = filename.split(".")
    for i in range(len(videoarray)):
        cv2.imwrite(filename + "_" + str(i) + ".png", videoarray[i])


def save_npz(videoarray, filename):
    filename, _ = filename.split(".")
    np.savez_compressed(filename, videoarray)


def convert_dataset(dataframe,
                    target_directory,
                    resize_shape=None,  # (width, height,)
                    grayscale=False,
                    downsampling_frames=40,
                    proprocessing_function=None,
                    save_as="video"):
    assert save_as in ["video", "images", "npz"], "Unrecognized argument for save_as: " + save_as
    target_directory = target_directory.rstrip("/\\") + "/"
    for _, row in dataframe.iterrows():
        os.makedirs(target_directory + str(row.category) + "/", exist_ok=True)
        vid = get_formatted_video(row.path, resize_shape, grayscale, downsampling_frames, proprocessing_function)
        if save_as == "video":
            save_video(vid, target_directory + str(row.category) + "/" + row.filename)
        if save_as == "images":
            save_images(vid, target_directory + str(row.category) + "/" + row.filename)
        if save_as == "npz":
            save_npz(vid, target_directory + str(row.category) + "/" + row.filename)


def convert_optflow_dataset(dataframe, target_directory, save_as="npz", stack_size=10):
    target_directory = target_directory.rstrip("/\\") + "/"
    assert save_as in ["npz"], "Unrecognized argument for save_as: " + save_as
    for _, row in dataframe.iterrows():
        os.makedirs(target_directory + str(row.category) + "/", exist_ok=True)
        vid = load_video(row.path)
        if save_as == "npz":
            optflow = calc_stacked_optical_flow(vid, stack_size)
            optflow = optflow.astype(int)
            save_npz(optflow, target_directory + str(row.category) + "/" + row.filename)


# simple dense optflow vector with horizontal and vertical component
def calc_optical_flow(frame1, frame2):
    assert (frame1.shape == frame2.shape), "Frames must have the same shape."
    # grayscale if necessary
    if frame1.shape[2] == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(frame1, frame2, None)
    return flow


def calc_stacked_optical_flow(video, stack_size=10):
    stack = []
    for i in range(len(video) - stack_size):
        frame = []
        for l in range(i + 1, i + 1 + stack_size):
            frame.append(calc_optical_flow(video[i], video[l]))
        stack.append(np.dstack(frame))
    return np.array(stack)


#############
# Attention #
#############

# combine multiple attention maps to a single overlay of fixed size
def combine_attention(attention, size=(224, 224)):
    # convert tf.Tensor to numpy, so cv2 can work with it
    for i in range(0, len(attention)):
        if isinstance(attention[i], tf.Tensor):
            attention[i] = attention[i].numpy()

    combined_attention = cv2.resize(attention[0], dsize=size)
    for i in range(1, len(attention)):
        # min-max scaling
        attention[i] = (attention[i] - np.min(attention[i]))/(np.max(attention[i]) - np.min(attention[i]))
        # combine
        combined_attention += cv2.resize(attention[i], dsize=size)
    combined_attention /= len(attention)
    return combined_attention


# display an image along with it's attention maps
def display_attention_maps(image, attention, rescale_image=True, cmap="inferno"):
    attention_count = len(attention)
    fig, ax = plt.subplots(1, attention_count + 1, figsize=(20, 20))
    # the source-image range will often be [-1, 1] and needs to be [0, 1] for display
    if rescale_image:
        ax[0].imshow(image / 2 + 0.5)
    else:
        ax[0].imshow(image)
    for i in range(attention_count):
        ax[i + 1].imshow(attention[i], cmap=plt.get_cmap(cmap))
    plt.show()


# combine image and overlay to a single image
def overlay_attention(image, overlay, rescale_image=True, cmap='inferno'):
    if rescale_image:
        image = image/2 + 0.5
    # rescale to [0, 1]
    if overlay.max() != 0:
        overlay = overlay / overlay.max()
    # grayscale to rgba heatmap
    colormap = plt.get_cmap(cmap)
    heatmap = colormap(overlay)
    # slice the alpha channel: rgba -> rgb
    heatmap = heatmap[:, :, :3]
    # from float64 to float 32 / safeguard types
    image = image.astype("float32")
    heatmap = heatmap.astype("float32")

    combined_image = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)
    return combined_image


def display_attention_batch(model, x, use_attention=False, CAM_layers=None, cmap='inferno'):
    attention = []
    if use_attention:
        extractor = AttentionModels.get_attention_extractor(model)
        att_list = extractor(x)
        attention = attention + att_list[1:]
    if CAM_layers is not None:
        for layer in CAM_layers:
            cbam_attention = get_gradcam_attention(x, model, layer)
            attention = attention + [cbam_attention]

    for i in range(x.shape[0]):
        attention_slice = []
        for a in attention:
            attention_slice.append(a[i])
        overlay = combine_attention(attention_slice, size=x.shape[1:3])
        combined_image = overlay_attention(x[i], overlay, cmap=cmap)
        display_attention_maps(x[i], [combined_image] + attention_slice, cmap=cmap)


def display_lime_batch(model, x, hide_rest=False):
    explainer = lime_image.LimeImageExplainer()
    for i in range(x.shape[0]):
        explanation = explainer.explain_instance(
            x[i].astype('double'),
            model.predict, top_labels=5,
            hide_color=0,
            num_samples=1000)

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=hide_rest)

        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()


############
# Grad-CAM #
############

def get_gradcam_attention(model_inputs, model, layer_name=None, layer_type=None):
    '''
    Adaption of https://keras.io/examples/vision/grad_cam/

    Args:
        model_inputs: input data, i.e. batch of images
        model: 2D-CNN Model
        layer_name: name of the layer the Grad-CAM is to be extracted from
        layer_type: type of the layer the Grad-CAM is to be extracted from

    Returns:

    '''
    # collect the observed cnn-layers
    # TODO by layer_type
    observed_layer = model.get_layer(layer_name)

    grad_model = Model(inputs=model.inputs, outputs=[model.output, observed_layer.output])

    with tf.GradientTape() as tape:
        preds, layer_output = grad_model(model_inputs)
        pred_index = tf.argmax(preds, axis=1)
        class_channel = []
        for i in range(len(pred_index)):
            class_channel.append(preds[i, pred_index[i]])

    gradients = tape.gradient(class_channel, layer_output)
    mean_gradients = tf.reduce_mean(gradients, axis=(1, 2))

    heatmaps = []
    for i in range(layer_output.shape[0]):
        weighted_activation = tf.matmul(layer_output[i], mean_gradients[i][..., tf.newaxis])
        weighted_activation = tf.squeeze(weighted_activation)
        weighted_activation = tf.maximum(weighted_activation, 0) / tf.math.reduce_max(weighted_activation)
        heatmaps.append(weighted_activation.numpy())

    return np.array(heatmaps)
