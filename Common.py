import os
import os.path
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


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
