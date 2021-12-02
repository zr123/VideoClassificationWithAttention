import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from VCWA import Common
import random
import VCWA.CompressedNumpy as cnp
from tqdm import tqdm


class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 dataframe,
                 target_size,
                 preprocessing_function,  # tf.keras.applications.resnet_v2.preprocess_input,
                 optflow=False,
                 batch_size=4,
                 shape_format="video",
                 single_frame=False,
                 rotation_range=None,
                 shear_range=None,
                 zoom_range=None,
                 horizontal_flip=False):
        self.dataframe = dataframe[["path", "category"]].copy().reset_index()
        self.optflow = optflow
        if optflow:
            self.dataframe["optflow_path"] = dataframe.optflow_path
        self.num_classes = self.dataframe.category.nunique()
        self.batch_size = batch_size
        self.img_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.preprocessing_function = preprocessing_function
        self.target_size = target_size
        self.n = len(self.dataframe)
        self.length = self.n // self.batch_size
        assert shape_format in ["video", "images"], "Unexpected argument for shape_format: " + shape_format
        self.shape_format = shape_format
        self.single_frame = single_frame
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        print(f"Found {self.n} videos belonging to {self.num_classes} classes.")
        self.on_epoch_end()

    def on_epoch_end(self):
        # shuffle
        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        x_batch_vid, y_batch = self.get_batch(index)
        if self.optflow:
            x_batch_optflow = self.get_x_batch_optflow(index)
            return [x_batch_vid, x_batch_optflow], y_batch
        else:
            return x_batch_vid, y_batch

    def get_batch(self, index):
        x_batch_vid = []
        y_batch = []
        for i in range(index * self.batch_size, self.batch_size * index + self.batch_size):
            if i == self.n:
                break
            x = self.get_x(self.dataframe.path[i])
            x_batch_vid.append(x)
            y = to_categorical(self.dataframe.category[i], num_classes=self.num_classes)
            if self.shape_format == "images":
                y = np.expand_dims(y, axis=0)
                y = np.repeat(y, x.shape[0], axis=0)
            y_batch.append(y)
        if self.shape_format == "video":
            x_batch_vid = np.array(x_batch_vid)
        if self.shape_format == "images":
            x_batch_vid = np.vstack(x_batch_vid)
        return x_batch_vid, np.vstack(y_batch)

    def get_x(self, path):
        x = self.load_video(path)
        x = self.format_video(x)
        if self.single_frame:
            x = self.get_single_frame(x)
        return x

    @staticmethod
    def get_single_frame(x):
        frame_count = x.shape[0]
        random_frame = random.randint(0, frame_count - 1)
        frame = x[random_frame].copy()
        del x
        frame = np.expand_dims(frame, axis=0)
        return frame

    def get_x_batch_optflow(self, index):
        x_batch_optflow = []
        for i in range(index, index + self.batch_size):
            if i == self.n:
                break
            x = self.get_x(self.dataframe.optflow_path[i])
            x_batch_optflow.append(x)

        if self.shape_format == "video":
            x_batch_optflow = np.array(x_batch_optflow)
        if self.shape_format == "images":
            x_batch_optflow = np.vstack(x_batch_optflow)
        return x_batch_optflow

    def __len__(self):
        return self.length

    # def load_compressed_dataset(self):
    #     self.data = cnp.Array()
    #     for i in tqdm(range(self.n)):
    #         self.data.append(Common.load_video(self.dataframe.path[i]))

    def load_video(self, path):
        file_extension = path.split(".")[-1]
        if file_extension == "npz":
            with np.load(path, allow_pickle=True) as npz:
                video = npz["arr_0"]
        if file_extension in ["avi", "mp4"]:
            video = Common.load_video(path)
        return video

    def format_video(self, video):
        formatted_video = []
        for frame in video:
            frame = self.format_frame(frame)
            formatted_video.append(frame)
        return np.array(formatted_video)

    def format_frame(self, frame):
        transform_params = self.get_random_transform_params()
        frame = self.img_datagen.apply_transform(frame, transform_params)
        frame = tf.keras.preprocessing.image.smart_resize(frame, self.target_size)
        if self.preprocessing_function is not None:
            frame = self.preprocessing_function(frame)
        return frame

    def get_random_transform_params(self):
        theta, shear = 0, 0
        zx, zy = 1, 1
        if self.rotation_range is not None:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        if self.shear_range is not None:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        if self.zoom_range is not None:
            zx, zy = np.random.uniform(1-self.zoom_range, 1.0, 2)
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        transform_params = {'theta': theta,
                            'shear': shear,
                            'zx': zx,
                            'zy': zy,
                            'flip_horizontal': flip_horizontal}
        return transform_params
