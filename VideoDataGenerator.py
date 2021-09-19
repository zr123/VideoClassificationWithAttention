import tensorflow as tf
import Common
import numpy as np
from tensorflow.keras.utils import to_categorical


class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 path,
                 y,
                 num_classes,
                 optflow_path=None,
                 batch_size=4,
                 target_size=None,
                 shear_range=None,
                 zoom_range=None,
                 horizontal_flip=False,
                 preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input):
        self.path = path
        self.y = y
        self.num_classes = num_classes
        self.optflow_path = optflow_path
        self.batch_size = batch_size
        self.img_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.preprocessing_function=preprocessing_function
        self.transform_params = {
            "shear": shear_range,
            "zx": zoom_range,
            "zy": zoom_range,
            "flip_horizontal": horizontal_flip
        }
        self.target_size = target_size
        self.n = len(self.path)
        self.length = self.n // self.batch_size

    def on_epoch_end(self):
        # shuffle?
        pass

    # /from directory & from
    def __getitem__(self, index):
        X_batch_vid = []
        X_batch_optflow = []
        y_batch = []
        for i in range(index, index + self.batch_size):
            if i == self.n:
                break
            X_batch_vid.append(self.weasel(self.path[i], "video"))
            X_batch_optflow.append(self.weasel(self.optflow_path[i], "npz"))
            y_batch.append(to_categorical(self.y[i], num_classes=self.num_classes))
        return [np.array(X_batch_vid), np.array(X_batch_optflow)], np.vstack(y_batch)

    def __len__(self):
        return self.length

    def weasel(self, path, format="video"):
        if format == "video":
            vid = Common.load_video(path)
        if format == "npz":
            npz = np.load(path, allow_pickle=True)
            vid = npz["arr_0"]
            npz.close()
        formatted_vid = []
        for frame in vid:
            frame = self.format_frame(frame)
            formatted_vid.append(frame)
        return np.array(formatted_vid)

    def format_frame(self, frame):
        frame = self.img_datagen.apply_transform(frame, self.transform_params)
        frame = tf.keras.preprocessing.image.smart_resize(frame, self.target_size)
        frame = self.preprocessing_function(frame)
        return frame
