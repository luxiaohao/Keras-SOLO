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

from model.neck import FPN
from model.head import DecoupledSOLOHead


class SOLO(object):
    def __init__(self, backbone, neck, head):
        super(SOLO, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def __call__(self, x, eval):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x, eval)
        return x




