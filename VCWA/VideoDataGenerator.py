import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from VCWA import Common


class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 path,
                 y,
                 num_classes,
                 target_size,
                 optflow_path=None,
                 batch_size=4,
                 preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
                 shape_format="video"):
                 #shear_range = None,
                 #zoom_range = None,
                 #horizontal_flip = False
        self.path = path
        self.y = y
        self.num_classes = num_classes
        self.optflow_path = optflow_path
        self.batch_size = batch_size
        self.img_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.preprocessing_function = preprocessing_function
        self.target_size = target_size
        self.n = len(self.path)
        self.length = self.n // self.batch_size
        assert shape_format in ["video", "images"], "Unexpected argument for shape_format: " + shape_format
        self.shape_format = shape_format
        # TODO: randomize transformation
        #self.transform_params = {}
        #if shear_range != None:
        #    self.transform_params.update({"shear": shear_range})
        #if zoom_range != None:
        #    self.transform_params.update({"zx": 1-zoom_range, "zy": 1-zoom_range})
        #if horizontal_flip != False:
        #    self.transform_params.update({"flip_horizontal": horizontal_flip})

    def on_epoch_end(self):
        # shuffle?
        pass

    def __getitem__(self, index):
        X_batch_vid, y_batch = self.get_batch(index)
        if self.optflow_path is not None:
            X_batch_optflow = self.get_x_batch_optflow(index)
            return [X_batch_vid, X_batch_optflow], y_batch
        else:
            return X_batch_vid, y_batch

    def get_batch(self, index):
        X_batch_vid = []
        y_batch = []
        for i in range(index, index + self.batch_size):
            if i == self.n:
                break
            x = self.load_and_format_video(self.path[i])
            X_batch_vid.append(x)
            y = to_categorical(self.y[i], num_classes=self.num_classes)
            if self.shape_format == "images":
                y = np.expand_dims(y, axis=0)
                y = np.repeat(y, x.shape[0], axis=0)
            y_batch.append(y)
        if self.shape_format == "video":
            X_batch_vid = np.array(X_batch_vid)
        if self.shape_format == "images":
            X_batch_vid = np.vstack(X_batch_vid)
        return X_batch_vid, np.vstack(y_batch)

    def get_x_batch_optflow(self, index):
        X_batch_optflow = []
        for i in range(index, index + self.batch_size):
            if i == self.n:
                break
            X_batch_optflow.append(self.load_and_format_video(self.optflow_path[i]))
        if self.shape_format == "video":
            X_batch_optflow = np.array(X_batch_optflow)
        if self.shape_format == "images":
            X_batch_optflow = np.vstack(X_batch_optflow)
        return X_batch_optflow

    def __len__(self):
        return self.length

    def load_and_format_video(self, path):
        file_extension = path.split(".")[-1]
        if file_extension == "npz":
            npz = np.load(path, allow_pickle=True)
            vid = npz["arr_0"]
            npz.close()
        if file_extension in ["avi", "mp4"]:
            vid = Common.load_video(path)

        formatted_vid = []
        for frame in vid:
            frame = self.format_frame(frame)
            formatted_vid.append(frame)
        return np.array(formatted_vid)

    def format_frame(self, frame):
        # TODO: randomize transform params
        # frame = self.img_datagen.apply_transform(frame, self.transform_params)
        frame = tf.keras.preprocessing.image.smart_resize(frame, self.target_size)
        if self.preprocessing_function is not None:
            frame = self.preprocessing_function(frame)
        return frame
