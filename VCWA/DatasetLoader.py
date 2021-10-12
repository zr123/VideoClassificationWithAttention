import tensorflow as tf
import numpy as np
from VCWA import Common


class DatasetLoader(tf.keras.utils.Sequence):

    def __init__(self,
                 X,
                 y,
                 batch_size=16,
                 resize_shape=None,
                 grayscale=False,
                 downsampling_frames=None,
                 proprocessing=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.grayscale = grayscale
        self.downsampling_frames = downsampling_frames
        self.proprocessing = proprocessing
        self.n = len(self.X)
        self.length = len(self.X) // batch_size

    def on_epoch_end(self):
        # shuffle?
        pass

    def __getitem__(self, index):
        X_batch = []
        y_batch = []
        for i in range(index, index + self.batch_size):
            if i == self.n:
                break
            X_batch.append(Common.get_formatted_video(self.X[i], self.resize_shape, self.grayscale, self.downsampling_frames, self.proprocessing))
            y_batch.append(self.y[i])
        return np.array(X_batch), np.vstack(y_batch)

    def __len__(self):
        return self.length
