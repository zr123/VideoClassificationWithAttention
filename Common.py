import os
import os.path
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def evaluate_dataset(path="D:\datasets\hmdb51_org", shuffle=False, random_state=42):
    df = pd.DataFrame()
    for path, directories, files in os.walk(path):
        for f in files:
            path = path.replace("/", "\\")
            df = df.append({
                "path": path + "\\" + f,
                "filename": f,
                "category": path.split("\\")[-1]},
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


def get_formatted_video(path, resize_shape=None, grayscale=False, downsampling_frames=None, proprocessing_function=None):
    video = load_video(path)
    if resize_shape is not None:
        video = resize_video(video, resize_shape)
    if downsampling_frames is not None:
        video = downsample_video(video, downsampling_frames)
    if grayscale:
        video = grayscale_video(video)
        video = video.reshape(video.shape[0:3] + (1,))
    if proprocessing_function is not None:
        video = proprocessing_function(video)
    return video


def save_video(videoarray, filename):
    frames, height, width = (videoarray.shape)[0:3]
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
    for _, row in dataframe.iterrows():
        os.makedirs(target_directory + str(row.category) + "/", exist_ok=True)
        vid = get_formatted_video(row.path, resize_shape, grayscale, downsampling_frames, proprocessing_function)
        if save_as == "video":
            save_video(vid, target_directory + str(row.category) + "/" + row.filename)
        if save_as == "images":
            save_images(vid, target_directory + str(row.category) + "/" + row.filename)
        if save_as == "npz":
            save_npz(vid, target_directory + str(row.category) + "/" + row.filename)


def convert_optflow_dataset(dataframe, target_directory, save_as="npz", L=10):
    assert save_as in ["npz"], "Unrecognized argument for save_as: " + save_as
    for _, row in dataframe.iterrows():
        os.makedirs(target_directory + str(row.category) + "/", exist_ok=True)
        vid = load_video(row.path)
        if save_as == "npz":
            optflow = calcStackedOpticalFlow(vid, L)
            save_npz(optflow, target_directory + str(row.category) + "/" + row.filename)


# simple dense optflow vector with horizontal and vertical component
def calcOpticalFlow(frame1, frame2):
    assert (frame1.shape == frame2.shape), "Frames must have the same shape."
    # grayscale if necessary
    if frame1.shape[2] == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# TODO: wie haben die das in dem Paper gemacht?
def calcStackedOpticalFlow(video, stack_size_L=10):
    stack = []
    for i in range(len(video) - stack_size_L):
        frame = []
        for l in range(i + 1, i + 1 + stack_size_L):
            frame.append(calcOpticalFlow(video[i], video[l]))
        stack.append(np.dstack(frame))
    return np.array(stack)


# combine multiple attention maps to a single overlay of fixed size
def combine_attention(attention, size=(224, 224)):
    # convert tf.Tensor to numpy, so cv2 can work with it
    for i in range(0, len(attention)):
        if isinstance(attention[i], tf.Tensor):
            attention[i] = attention[i].numpy()

    combined_attention = cv2.resize(attention[0], dsize=size)
    for i in range(1, len(attention)):
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
    overlay = overlay / overlay.max()
    # grayscale to rgba heatmap
    colormap = plt.get_cmap(cmap)
    heatmap = colormap(overlay)
    # slice the alpha channel: rgba -> rgb
    heatmap = heatmap[:, :, :3]
    # from float64 to float 32
    heatmap = heatmap.astype("float32")

    combined_image = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)
    return combined_image
