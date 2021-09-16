import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D, Activation
from tensorflow.keras.activations import sigmoid
from tensorflow.python.keras.applications import resnet
import AttentionCommon


def create_residual_attention_module(x, filters, p=1, t=2, r=1, residual_block_fn=resnet.block2, shortcuts=0,
                                     name=None, attention_function="sigmoid"):
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
    """
    assert attention_function in ["softmax", "sigmoid", "pseudo-softmax"], "Unexpected attention_function argument."
    # p: pre-processing
    for i in range(p):
        x = residual_block_fn(x, filters, name=name + "_preblock" + str(i))
    # t: trunk branch
    trunk = x
    for i in range(t):
        trunk = residual_block_fn(trunk, filters, name=name + "_trunkblock" + str(i))
    # mask branch
    mask = x
    mask = create_mask_branch(mask, filters, r, residual_block_fn, shortcuts, name, attention_function)
    # fusion
    x = (trunk * mask) + trunk
    # p: post-processing
    for i in range(p):
        x = residual_block_fn(x, filters, name=name + "_postblock" + str(i))
    return x


def create_mask_branch(x, filters, r, residual_block_fn, shortcuts, name, attention_function):
    # pre-block
    x = MaxPooling2D()(x)
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_preblock" + str(i))
    # inner blocks & shortcutting
    x = create_mask_branch_inner_shortcuts(x, filters, r, residual_block_fn, shortcuts, name)
    # post-block
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_postblock" + str(i))
    x = UpSampling2D()(x)
    x = Conv2D(filters * 4, 1)(x)
    x = Conv2D(filters * 4, 1)(x)
    if attention_function == "softmax":
        x = AttentionCommon.softmax2d(x, name=name + "_ResidualAttention")
    if attention_function == "sigmoid":
        x = Activation(sigmoid, name=name + "_ResidualAttention")(x)
    if attention_function == "pseudo-softmax":
        x = AttentionCommon.pseudo_softmax2d(x, name=name + "_ResidualAttention")
    return x


def create_mask_branch_inner_shortcuts(x, filters, r, residual_block_fn, shortcuts, name):
    """ Creates the nested inner parts of the mask branch.
    """
    if shortcuts == 0:
        return x
    shortcut = x 
    x = MaxPooling2D()(x)
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_inner" + str(shortcuts) + "_preblock" + str(i))
    x = create_mask_branch_inner_shortcuts(x, filters, r, residual_block_fn, shortcuts - 1, name)
    for i in range(r):
        x = residual_block_fn(x, filters, name=name + "_mask_inner" + str(shortcuts) + "_postblock" + str(i))
    x = UpSampling2D()(x)
    x = x + shortcut
    return x
