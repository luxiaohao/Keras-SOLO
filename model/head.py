#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import tensorflow as tf
import keras.layers as layers
from model.custom_layers import Resize


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



class DecoupledSOLOHead(object):
    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.4,
                 num_grids=None,
                 cate_down_pos=0,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(DecoupledSOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        a = layers.Nor
        self.ins_convs_x = nn.ModuleList()
        self.ins_convs_y = nn.ModuleList()
        self.cate_convs = nn.ModuleList()



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
        ins_feat_x, ins_feat_y = layers.Lambda(concat_coord)(new_feats[i])
        # a = forward_single(new_feats[i], i, eval, upsampled_size)
    return new_feats






