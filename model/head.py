#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import keras
import tensorflow as tf
import keras.layers as layers
from keras import backend as K
from keras.engine.topology import Layer

from model.resnet import conv2d_unit


def DecoupledSOLOHead(xs, out_channels, start_level, num_outs, add_extra_convs=False):
    return 1






