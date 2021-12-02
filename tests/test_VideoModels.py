from pathlib import Path
import pytest
import tensorflow as tf
import numpy as np
from VCWA import VideoModels
from VCWA import AttentionModels



################
# Video Models #
################

@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_2DCNN_MLP(classes):
    model = VideoModels.create_2DCNN_MLP((40, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_3DCNN(classes):
    model = VideoModels.create_3DCNN((40, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


def test_assemble_lstm():
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    backbone = tf.keras.applications.MobileNetV2()
    lstm = VideoModels.assemble_lstm(backbone, classes=51)
    tf.debugging.assert_shapes([(lstm.output, (None, 51))])
    lstm.predict(dummy_vid)


def test_assemble_TwoStreamModel_MobileNetV2():
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    dummy_optflow = np.zeros((1, 15, 224, 224, 20))
    vid_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, classes=10, weights=None)
    optflow_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 20), include_top=True, classes=10, weights=None)
    two_stream_model = VideoModels.assemble_TwoStreamModel(vid_model, optflow_model, 51, fusion="average", recreate_top=True)
    tf.debugging.assert_shapes([(two_stream_model.output, (None, 51))])
    two_stream_model.predict([dummy_vid, dummy_optflow])


def test_assemble_TwoStreamModel_CBAM():
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    dummy_optflow = np.zeros((1, 15, 224, 224, 20))
    vid_model = AttentionModels.create_CBAM_MobileNetV2(input_shape=(224, 224, 3), classes=10)
    optflow_model = AttentionModels.create_CBAM_MobileNetV2(input_shape=(224, 224, 20), classes=10)
    two_stream_model = VideoModels.assemble_TwoStreamModel(vid_model, optflow_model, 51, fusion="average", recreate_top=True)
    tf.debugging.assert_shapes([(two_stream_model.output, (None, 51))])
    two_stream_model.predict([dummy_vid, dummy_optflow])


#####################
# Utility-functions #
#####################

def test_get_twostream_attention():
    vid_model = AttentionModels.create_ResidualAttention_MobileNetV2((224, 224, 3), classes=51)
    optflow_model = AttentionModels.create_ResidualAttention_MobileNetV2((224, 224, 3), classes=51)
    two_stream_model = VideoModels.assemble_TwoStreamModel(vid_model, optflow_model, 51, fusion="average", recreate_top=True)
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    attention = VideoModels.get_twostream_attention(dummy_vid[0], two_stream_model, include_input=False)
    assert attention.shape == (20, 224, 224, 3)


def test_get_twostream_attention_with_input():
    vid_model = AttentionModels.create_ResidualAttention_MobileNetV2((224, 224, 3), classes=51)
    optflow_model = AttentionModels.create_ResidualAttention_MobileNetV2((224, 224, 3), classes=51)
    two_stream_model = VideoModels.assemble_TwoStreamModel(vid_model, optflow_model, 51, fusion="average", recreate_top=True)
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    attention = VideoModels.get_twostream_attention(dummy_vid[0], two_stream_model)
    assert attention.shape == (20, 224, 448, 3)


def test_video_to_gif():
    dummy_vid = np.zeros((20, 224, 224, 3))
    VideoModels.video_to_gif(dummy_vid, "./tests/data/attention.gif")
    assert Path("./tests/data/attention.gif").exists
