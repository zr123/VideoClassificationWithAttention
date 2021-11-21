import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import sigmoid
from TF_modification.resatt_mobilenet_v2 import _inverted_res_block
from VCWA import Common


def residual_block_helper(x, filters, name):
    return _inverted_res_block(
        x,
        filters=filters,
        alpha=1.0,
        stride=1,
        expansion=6,
        block_id=name)


def create_residual_attention_module(
        x,
        filters,
        p=1,
        t=2,
        r=1,
        residual_block_fn=residual_block_helper,
        shortcuts=0,
        name=None,
        attention_function="sigmoid"):
    """Residual attention module from Residual Attention Network for Image Classification by Wang et al.

        Arguments:
            x: input tensor
            filters: integer, convolution-filters of the bottleneck layer.
            p: default 1, number of pre/post-processing Residual Units
            t: default 2, number of Residual Units in the trunk branch
            r: default 1, number of Residual Units between adjacent pooling layer in the mask branch
            residual_block_fn: helper function to construct a residual block.
            shortcuts: number of inner shortcuts in the mask branch
            name: name of the block
            attention_function: the function used to calculate mask, may be softmax, sigmoid or pseudo-softmax
    """
    assert attention_function in ["softmax", "sigmoid", "pseudo-softmax"], "Unexpected attention_function argument."
    # p: pre-processing
    for i in range(p):
        x = residual_block_fn(x, filters, name=name + "_preblock" + str(i))
    # mask branch
    mask = create_mask_branch(x, filters, r, residual_block_fn, shortcuts, name, attention_function)
    # t: trunk branch (x becomes trunk)
    for i in range(t):
        x = residual_block_fn(x, filters, name=name + "_trunkblock" + str(i))
    # fusion
    x = layers.Add()([x, layers.Multiply()([x, mask])])
    # p: post-processing
    for i in range(p):
        x = residual_block_fn(x, filters, name=name + "_postblock" + str(i))
    return x


def create_mask_branch(x, filters, r, residual_block_fn, shortcuts, name, attention_function):
    # pre-block
    x = layers.MaxPooling2D()(x)
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_preblock" + str(i))
    # inner blocks & shortcutting
    x = create_mask_branch_inner_shortcuts(x, filters, r, residual_block_fn, shortcuts, name)
    # post-block
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_postblock" + str(i))
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    x = layers.Conv2D(filters, 1)(x)
    x = layers.Conv2D(filters, 1)(x)
    if attention_function == "softmax":
        x = Common.softmax2d(x, name=name + "_ResidualAttention")
    if attention_function == "sigmoid":
        x = layers.Activation(sigmoid, name=name + "_ResidualAttention")(x)
    if attention_function == "pseudo-softmax":
        x = Common.pseudo_softmax2d(x, name=name + "_ResidualAttention")
    return x


def create_mask_branch_inner_shortcuts(x, filters, r, residual_block_fn, shortcuts, name):
    """ Creates the nested inner parts of the mask branch.
    """
    if shortcuts == 0:
        return x
    shortcut = x 
    x = layers.MaxPooling2D()(x)
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_inner" + str(shortcuts) + "_preblock" + str(i))
    x = create_mask_branch_inner_shortcuts(x, filters, r, residual_block_fn, shortcuts - 1, name)
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_inner" + str(shortcuts) + "_postblock" + str(i))
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    x = tf.keras.layers.Add()([x, shortcut])
    return x
