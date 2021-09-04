import sys

# set path
sys.path.append('../VideoClassificationWithAttention')

import Models
import AttentionModels
import tensorflow as tf


def test_create_2DCNN_MLP():
	model = Models.create_2DCNN_MLP((40, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_create_3DCNN():
	model = Models.create_3DCNN((40, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_create_LSTM():
	model = Models.create_LSTM((None, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_create_Transfer_LSTM():
	model = Models.create_Transfer_LSTM((None, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_lstm_test():
	model = Models.lstm_test((None, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_create_TwoStreamModel():
	model = Models.create_TwoStreamModel((None, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_create_L2PA_ResNet50v2():
	model = AttentionModels.create_L2PA_ResNet50v2((None, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])


def test_create_AttentionGated_ResNet50v2():
	model = AttentionModels.create_AttentionGated_ResNet50v2((None, 224, 224, 3), num_classes=51)
	tf.debugging.assert_shapes([(model.output, (None, 51))])
