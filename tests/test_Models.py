import sys
import pytest

# set path
sys.path.append('../VideoClassificationWithAttention')

import Models
import AttentionModels
import tensorflow as tf


################
# Video Models #
################

@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_2DCNN_MLP(classes):
    model = Models.create_2DCNN_MLP((40, 224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_3DCNN(classes):
    model = Models.create_3DCNN((40, 224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_LSTM(classes):
    model = Models.create_LSTM((None, 224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_Transfer_LSTM(classes):
    model = Models.create_Transfer_LSTM((None, 224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_lstm_test(classes):
    model = Models.lstm_test((None, 224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [51, 101, 174])
def test_create_TwoStreamModel(classes):
    model = Models.create_TwoStreamModel((None, 224, 224, 3), (None, 224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


################
# Image Models #
################

@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_L2PA_ResNet50v2(classes):
    model = AttentionModels.create_L2PA_ResNet50v2((224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_AttentionGated_ResNet50v2(classes):
    model = AttentionModels.create_AttentionGated_ResNet50v2((224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_AttentionGatedGrid_ResNet50v2(classes):
    model = AttentionModels.create_AttentionGatedGrid_ResNet50v2((224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_ResidualAttention_ResNet50v2(classes):
    model = AttentionModels.create_ResidualAttention_ResNet50v2((224, 224, 3), num_classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])
