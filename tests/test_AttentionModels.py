from pathlib import Path
import pytest
import tensorflow as tf
import numpy as np
from VCWA import VideoModels
from VCWA import AttentionModels


################
# Sanity Tests #
################

@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_L2PA_MobileNetV2(classes):
    model = AttentionModels.create_L2PA_MobileNetV2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_AttentionGated_MobileNetV2(classes):
    model = AttentionModels.create_AttentionGated_MobileNetV2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_AttentionGatedGrid_MobileNetV2(classes):
    model = AttentionModels.create_AttentionGatedGrid_MobileNetV2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_ResidualAttention_MobileNetV2(classes):
    model = AttentionModels.create_ResidualAttention_MobileNetV2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


@pytest.mark.parametrize("classes", [10, 100, 1000])
def test_create_CBAM_ResNet50v2(classes):
    model = AttentionModels.create_CBAM_MobileNetV2((224, 224, 3), classes=classes)
    tf.debugging.assert_shapes([(model.output, (None, classes))])


#######################
# Forward Propagation #
#######################

def test_create_L2PA_MobileNetV2_forward():
    dummy_image = np.zeros((1, 224, 224, 3))
    model = AttentionModels.create_L2PA_MobileNetV2((224, 224, 3), classes=51)
    model.predict(dummy_image)


def test_create_AttentionGated_MobileNetV2_forward():
    dummy_image = np.zeros((1, 224, 224, 3))
    model = AttentionModels.create_AttentionGated_MobileNetV2((224, 224, 3), classes=51)
    model.predict(dummy_image)


def test_create_AttentionGatedGrid_MobileNetV2_forward():
    dummy_image = np.zeros((1, 224, 224, 3))
    model = AttentionModels.create_AttentionGatedGrid_MobileNetV2((224, 224, 3), classes=51)
    model.predict(dummy_image)


def test_create_ResidualAttention_MobileNetV2_forward():
    dummy_image = np.zeros((1, 224, 224, 3))
    model = AttentionModels.create_ResidualAttention_MobileNetV2((224, 224, 3), classes=51)
    model.predict(dummy_image)


def test_create_CBAM_ResNet50v2_forward():
    dummy_image = np.zeros((1, 224, 224, 3))
    model = AttentionModels.create_CBAM_MobileNetV2((224, 224, 3), classes=51)
    model.predict(dummy_image)


##########################
# Model Saving & Loading #
##########################

def test_create_L2PA_MobileNetV2_saving():
    dummy_x = np.zeros((1, 224, 224, 3))
    dummy_y = np.zeros((1, 51))
    #AttentionModels.create_L2PA_MobileNetV2().save("./tests/data/models/L2PA")
    model = tf.keras.models.load_model("./tests/data/models/L2PA")
    model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.fit(dummy_x, dummy_y)


def test_create_AttentionGated_MobileNetV2_saving():
    dummy_x = np.zeros((1, 224, 224, 3))
    dummy_y = np.zeros((1, 51))
    AttentionModels.create_AttentionGated_MobileNetV2().save("./tests/data/models/Gated")
    model = tf.keras.models.load_model("./tests/data/models/Gated")
    model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.fit(dummy_x, dummy_y)


def test_create_AttentionGatedGrid_MobileNetV2_saving():
    dummy_x = np.zeros((1, 224, 224, 3))
    dummy_y = np.zeros((1, 51))
    AttentionModels.create_AttentionGatedGrid_MobileNetV2().save("./tests/data/models/Grid")
    model = tf.keras.models.load_model("./tests/data/models/Grid")
    model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.fit(dummy_x, dummy_y)


def test_create_ResidualAttention_MobileNetV2_saving():
    dummy_x = np.zeros((1, 224, 224, 3))
    dummy_y = np.zeros((1, 51))
    AttentionModels.create_ResidualAttention_MobileNetV2().save("./tests/data/models/Res")
    model = tf.keras.models.load_model("./tests/data/models/Res")
    model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.fit(dummy_x, dummy_y)


def test_create_CBAM_ResNet50v2_saving():
    dummy_x = np.zeros((1, 224, 224, 3))
    dummy_y = np.zeros((1, 51))
    AttentionModels.create_CBAM_MobileNetV2().save("./tests/data/models/CBAM")
    model = tf.keras.models.load_model("./tests/data/models/CBAM")
    model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.fit(dummy_x, dummy_y)
