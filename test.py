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

from model.resnet import Resnet50
from model.solo import SOLO


num_classes = 80

inputs = layers.Input(shape=(None, None, 3))
inputs = layers.Input(shape=(416, 416, 3))
outs = SOLO(inputs, num_classes, use_dcn=False)
model = keras.models.Model(inputs=inputs, outputs=outs)
# model.summary()
# keras.utils.vis_utils.plot_model(model, to_file='solo.png', show_shapes=True)




