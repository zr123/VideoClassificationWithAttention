import sys
import pytest

# set path
sys.path.append('../VideoClassificationWithAttention')

import Models
import AttentionModels
import tensorflow as tf
import numpy as np


################
# Video Models #
################

@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_2DCNN_MLP(classes):
    model = Models.create_2DCNN_MLP((40, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_3DCNN(classes):
    model = Models.create_3DCNN((40, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_LSTM(classes):
    model = Models.create_LSTM((None, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_Transfer_LSTM(classes):
    model = Models.create_Transfer_LSTM((None, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_lstm_test(classes):
    model = Models.lstm_test((None, 224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_TwoStreamModel(classes):
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    dummy_optflow = np.zeros((1, 15, 224, 224, 20))
    model = Models.create_TwoStreamModel(
        input_shape=(None, 224, 224, 3),
        optflow_shape=(None, 224, 224, 20),
        classes=classes
    )
    tf.debugging.assert_shapes([(model.output, (None, classes))])
    model.predict([dummy_vid, dummy_optflow])


def test_assemble_TwoStreamModel():
    dummy_vid = np.zeros((1, 20, 224, 224, 3))
    dummy_optflow = np.zeros((1, 15, 224, 224, 20))
    vid_model = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), include_top=True, classes=10, weights=False)
    optflow_model = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 20), include_top=True, classes=10, weights=False)
    two_stream_model = Models.assemble_TwoStreamModel(vid_model, optflow_model, 51, fusion="average", recreate_top=True)
    tf.debugging.assert_shapes([(two_stream_model.output, (None, 51))])
    two_stream_model.predict([dummy_vid, dummy_optflow])


################
# Image Models #
################

@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_L2PA_ResNet50v2(classes):
    model = AttentionModels.create_L2PA_ResNet50v2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_AttentionGated_ResNet50v2(classes):
    model = AttentionModels.create_AttentionGated_ResNet50v2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_AttentionGatedGrid_ResNet50v2(classes):
    model = AttentionModels.create_AttentionGatedGrid_ResNet50v2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_ResidualAttention_ResNet50v2(classes):
    model = AttentionModels.create_ResidualAttention_ResNet50v2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_CBAM_ResNet50v2(classes):
    model = AttentionModels.create_CBAM_ResNet50v2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])
