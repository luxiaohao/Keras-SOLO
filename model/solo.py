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
from model.neck import FPN
from model.head import DecoupledSOLOHead


def SOLO(x, num_classes, use_dcn, out_channels=256, start_level=0, num_outs=5):
    x = Resnet50(x, use_dcn=use_dcn)
    x = FPN(x, out_channels, start_level, num_outs, add_extra_convs=False)
    x = DecoupledSOLOHead(x)
    return x




