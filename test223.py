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

import torch
import numpy as np

num_classes = 80


class ConcatCoord(Layer):
    def __init__(self):
        super(ConcatCoord, self).__init__()

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        ins_feat = x  # [N, h, w, c]
        # ins branch
        # concat coord
        # x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)   # [w, ]
        # y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)   # [h, ]

        batch_size = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        float_h = tf.cast(h, tf.float32)
        float_w = tf.cast(w, tf.float32)

        y_range = tf.range(float_h, dtype=tf.float32)     # [h, ]
        y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
        x_range = tf.range(float_w, dtype=tf.float32)     # [w, ]
        x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
        x_range = x_range[tf.newaxis, :]   # [1, w]
        y_range = y_range[:, tf.newaxis]   # [h, 1]
        x = tf.tile(x_range, [h, 1])     # [h, w]
        y = tf.tile(y_range, [1, w])     # [h, w]

        x = x[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
        y = y[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
        x = tf.tile(x, [batch_size, 1, 1, 1])   # [N, h, w, 1]
        y = tf.tile(y, [batch_size, 1, 1, 1])   # [N, h, w, 1]

        ins_feat_x = tf.concat([ins_feat, x], axis=-1)   # [N, h, w, c+1]
        ins_feat_y = tf.concat([ins_feat, y], axis=-1)   # [N, h, w, c+1]
        # aaa = 2.0 * aaa / h - 1.0
        # y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        # x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        # xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        # xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        # xy_grid = tf.cast(xy_grid, tf.float32)

        # y, x = torch.meshgrid(y_range, x_range)   # y [h, w]     x [h, w]
        # y = y.expand([ins_feat.shape[0], 1, -1, -1])   # [N, 1, h, w]
        # x = x.expand([ins_feat.shape[0], 1, -1, -1])   # [N, 1, h, w]
        # ins_feat_x = torch.cat([ins_feat, x], 1)    # [N, c+1, h, w]
        # ins_feat_y = torch.cat([ins_feat, y], 1)    # [N, c+1, h, w]
        # return [ins_feat_x, ins_feat_y]
        return ins_feat_x



def concat_coord(x):
    ins_feat = x  # [N, h, w, c]

    batch_size = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    float_h = tf.cast(h, tf.float32)
    float_w = tf.cast(w, tf.float32)

    y_range = tf.range(float_h, dtype=tf.float32)     # [h, ]
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = tf.range(float_w, dtype=tf.float32)     # [w, ]
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = x_range[tf.newaxis, :]   # [1, w]
    y_range = y_range[:, tf.newaxis]   # [h, 1]
    x = tf.tile(x_range, [h, 1])     # [h, w]
    y = tf.tile(y_range, [1, w])     # [h, w]

    x = x[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
    y = y[tf.newaxis, :, :, tf.newaxis]   # [1, h, w, 1]
    x = tf.tile(x, [batch_size, 1, 1, 1])   # [N, h, w, 1]
    y = tf.tile(y, [batch_size, 1, 1, 1])   # [N, h, w, 1]

    ins_feat_x = tf.concat([ins_feat, x], axis=-1)   # [N, h, w, c+1]
    ins_feat_y = tf.concat([ins_feat, y], axis=-1)   # [N, h, w, c+1]

    return [ins_feat_x, ins_feat_y]


all_pred_scores = layers.Input(name='all_pred_scores', shape=(None, None, 256))


ins_feat_x, ins_feat_y = layers.Lambda(concat_coord)(all_pred_scores)



a = np.zeros((8, 6, 4, 256), np.float32)



sess = K.get_session()
aaa00 = sess.run(ins_feat_x, feed_dict={all_pred_scores: a, })

# aaa01 = np.mean(aaa00)573 405   372 394
# aaa02 = np.var(aaa00)

print()


