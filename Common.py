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
    sampled_frames = np.arange(0, framecount, framecount / downsampling_frames)
    downsampled_video = [video[int(f)] for f in sampled_frames]
    return np.array(downsampled_video)


def grayscale_video(video):
    grayscaled_video = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video
    ]
    return np.array(grayscaled_video)


def get_formatted_video(path, resize_shape=None, grayscale=False, downsampling_frames=None, proprocessing=None):
    video = load_video(path)
    if resize_shape is not None:
        video = resize_video(video, resize_shape)
    if downsampling_frames is not None:
        video = downsample_video(video, downsampling_frames)
    if grayscale:
        video = grayscale_video(video)
        video = video.reshape(video.shape[0:3] + (1,))
    if proprocessing is not None:
        if proprocessing == "normalize":
            video = video / 255.0
            video = np.float32(video)
        elif proprocessing == "InceptionResNetV2":
            video = tf.keras.applications.inception_resnet_v2.preprocess_input(video)
        else:
            raise Exception("Unexpected argument for preprocessing")
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


def convert_dataset(dataframe, target_directory,
                    resize_shape=(128, 128),  # (width, height,)
                    grayscale=False,
                    downsampling_frames=40,
                    normalize=False):
    for _, row in dataframe.iterrows():
        os.makedirs(target_directory + str(row.category) + "/", exist_ok=True)
        vid = get_formatted_video(row.path, resize_shape, grayscale, downsampling_frames, normalize)
        save_video(vid, target_directory + str(row.category) + "/" + row.filename)


# simple dense optflow vector with horizontal and vertical component
def calcOpticalFlow(frame1, frame2):
    assert (frame1.shape == frame2.shape), "Frames must have the same shape."
    # grayscale if necessary
    if frame1.shape[2] == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def calcStackedOpticalFlow(video, stack_size_L=10):
    stack = []
    for i in range(len(video)):
        frame = []
        for l in range(i + 1, i + 1 + stack_size_L):
            if l >= len(video):
                frame.append(np.zeros((video.shape[1:3]) + (2,), dtype=np.float32))
            else:
                frame.append(calcOpticalFlow(video[i], video[l]))
        stack.append(np.dstack(frame))
    return np.array(stack)


# combine multiple attention maps to a single overlay of fixed size
def combine_attention(attention, size=(224, 224)):
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
