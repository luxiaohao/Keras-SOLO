#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import keras.layers as layers
from model.custom_layers import conv2d_unit


def FPN(xs, out_channels, start_level, num_outs, add_extra_convs=False):
    num_ins = len(xs)

    # build laterals
    laterals = []
    for i in range(num_ins):
        x = conv2d_unit(xs[i + start_level], out_channels, 1, strides=1, padding='valid', use_bias=False, bn=0, act=None)
        laterals.append(x)

    # build top-down path
    used_backbone_levels = len(laterals)
    for i in range(used_backbone_levels - 1, 0, -1):
        x = layers.UpSampling2D(2)(laterals[i])
        laterals[i - 1] = layers.add([laterals[i - 1], x])

    # build outputs
    # part 1: from original levels
    outs = []
    for i in range(used_backbone_levels):
        x = conv2d_unit(laterals[i], out_channels, 3, strides=1, padding='same', use_bias=False, bn=0, act=None)
        outs.append(x)
    # part 2: add extra levels
    if num_outs > len(outs):
        # use max pool to get more levels on top of outputs
        # (e.g., Faster R-CNN, Mask R-CNN)
        if not add_extra_convs:
            for i in range(num_outs - used_backbone_levels):
                x = layers.MaxPooling2D(pool_size=1, strides=2, padding='valid')(outs[-1])
                outs.append(x)
        # add conv layers on top of original feature maps (RetinaNet)
        else:
            pass
    return outs






