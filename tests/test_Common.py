import pytest
import numpy as np
from VCWA import Common
import tensorflow as tf


@pytest.mark.parametrize("original_frames,downsampling_frames", [(100, 40), (69, 25)])
def test_downsample_video(original_frames, downsampling_frames):
    dummyvideo = np.zeros((original_frames, 128, 128, 3), np.uint8)
    downsampled_video = Common.downsample_video(dummyvideo, downsampling_frames)
    assert(downsampled_video.shape == (downsampling_frames, 128, 128, 3))


@pytest.mark.parametrize("L", [1, 5, 10])
def test_calcStackedOpticalFlow(L):
    dummyvideo = np.zeros((40, 128, 128, 3), np.uint8)
    optflowstack = Common.calc_stacked_optical_flow(dummyvideo, L)
    assert(optflowstack.shape == (40-L, 128, 128, 2*L))


def test_evaluate_dataset():
    dataset = Common.evaluate_dataset("./tests/data/dummy_dataset")
    assert dataset.size == 30
    assert dataset.category.nunique() == 3
    assert dataset.groupby("category").count().filename.tolist() == [3, 5, 2]


@pytest.mark.parametrize("split_no", [1, 2, 3])
def test_get_hmdb51_split(split_no):
    df = Common.get_hmdb51_split("./tests/data/hmdb51_org_splits", split_no=split_no)
    assert df.size == 13532


@pytest.mark.parametrize("split_no, train_size, test_size", [(1, 9537, 3783), (2, 9586, 3734), (3, 9624, 3696)])
def test_get_ucf101_split(split_no, train_size, test_size):
    df = Common.get_ucf101_split("./tests/data/ucfTrainTestlist", split_no=split_no)
    assert df.split.size == 13_320
    assert df.split[df.split == 0].size == train_size
    assert df.split[df.split == 1].size == test_size


# test extracting Grad-CAM attention from ResNet50v2
def test_get_gradcam_attention():
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        './tests/data/dummy_imagenette',
        target_size=(224, 224),
        batch_size=8)
    basenet = tf.keras.applications.ResNet50V2(weights=None)
    x, y = test_generator.__getitem__(0)
    heatmaps = Common.get_gradcam_attention(x, basenet, layer_name="conv5_block3_3_conv")
    assert heatmaps.shape == (8, 7, 7)
