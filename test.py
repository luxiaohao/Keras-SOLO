#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================

import numpy as np
import keras
import tensorflow as tf
import keras.layers as layers
from keras import backend as K
from keras.engine.topology import Layer

from model.resnet import Resnet50
from model.solo import SOLO


# inputs = layers.Input(shape=(None, None, 3))
inputs = layers.Input(shape=(416, 416, 3))

eval=True
outs = SOLO(inputs, use_dcn=False, eval=eval)
model = keras.models.Model(inputs=inputs, outputs=outs)
model.summary()
keras.utils.vis_utils.plot_model(model, to_file='solo.png', show_shapes=True)

# aa = np.zeros((8, 416, 416, 3), np.float32)
# xxx = model.predict(aa)

print()




