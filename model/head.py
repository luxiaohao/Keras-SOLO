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

class Resize(Layer):
    def __init__(self, h, w, method):
        super(Resize, self).__init__()
        self.h = h
        self.w = w
        self.method = method
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.h, self.w, input_shape[3])
    def call(self, x):
        if self.method == 'BICUBIC':
            m = tf.image.ResizeMethod.BICUBIC
        elif self.method == 'NEAREST_NEIGHBOR':
            m = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif self.method == 'BILINEAR':
            m = tf.image.ResizeMethod.BILINEAR
        elif self.method == 'AREA':
            m = tf.image.ResizeMethod.AREA
        a = tf.image.resize_images(x, [self.h, self.w], method=m)
        return a



def forward_single(x, idx, eval, upsampled_size):
    ins_feat = x
    cate_feat = x
    # ins branch
    # concat coord
    x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
    y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([ins_feat.shape[0], 1, -1, -1])
    x = x.expand([ins_feat.shape[0], 1, -1, -1])
    ins_feat_x = torch.cat([ins_feat, x], 1)
    ins_feat_y = torch.cat([ins_feat, y], 1)

def DecoupledSOLOHead(feats, num_grids=[40, 36, 24, 16, 12], eval=True):
    # DecoupledSOLOHead都是这样，一定有5个张量，5个张量的strides=[8, 8, 16, 32, 32]，所以先对首尾张量进行插值。
    new_feats = [Resize(tf.shape(feats[1])[1], tf.shape(feats[1])[2], 'BILINEAR')(feats[0]),
                 feats[1],
                 feats[2],
                 feats[3],
                 Resize(tf.shape(feats[3])[1], tf.shape(feats[3])[2], 'BILINEAR')(feats[4])]
    # for k in new_feats:
    #     print(k.shape)
    featmap_sizes = [tf.shape(featmap)[1:3] for featmap in new_feats]
    upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)   # stride=4

    for i in range(len(num_grids)):
        a = forward_single(new_feats[i], i, eval, upsampled_size)
    return new_feats






