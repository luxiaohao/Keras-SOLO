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
import keras
import keras.layers as layers
from model.custom_layers import Resize, GroupNormalization


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
                 num_classes=80,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 sigma=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None):
        super(DecoupledSOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        # self.loss_cate = build_loss(loss_cate)
        # self.ins_loss_weight = loss_ins['loss_weight']
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.ins_convs_x = []
        self.ins_convs_y = []
        self.cate_convs = []

        for i in range(self.stacked_convs):
            conv2d_1 = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False)
            gn_1 = GroupNormalization(num_groups=32)
            relu_1 = layers.advanced_activations.ReLU()
            self.ins_convs_x.append(conv2d_1)
            self.ins_convs_x.append(gn_1)
            self.ins_convs_x.append(relu_1)

            conv2d_2 = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False)
            gn_2 = GroupNormalization(num_groups=32)
            relu_2 = layers.advanced_activations.ReLU()
            self.ins_convs_y.append(conv2d_2)
            self.ins_convs_y.append(gn_2)
            self.ins_convs_y.append(relu_2)

            conv2d_3 = layers.Conv2D(self.seg_feat_channels, 3, padding='same', strides=1, use_bias=False)
            gn_3 = GroupNormalization(num_groups=32)
            relu_3 = layers.advanced_activations.ReLU()
            self.cate_convs.append(conv2d_3)
            self.cate_convs.append(gn_3)
            self.cate_convs.append(relu_3)

        self.dsolo_ins_list_x = []
        self.dsolo_ins_list_y = []
        for seg_num_grid in self.seg_num_grids:
            conv2d_1 = layers.Conv2D(seg_num_grid, 3, padding='same', strides=1, use_bias=True)
            self.dsolo_ins_list_x.append(conv2d_1)
            conv2d_2 = layers.Conv2D(seg_num_grid, 3, padding='same', strides=1, use_bias=True)
            self.dsolo_ins_list_y.append(conv2d_2)
        self.dsolo_cate = layers.Conv2D(self.num_classes, 3, padding='same', strides=1, use_bias=True)

    def __call__(self, feats, eval):
        # DecoupledSOLOHead都是这样，一定有5个张量，5个张量的strides=[8, 8, 16, 32, 32]，所以先对首尾张量进行插值。
        new_feats = [Resize(tf.shape(feats[1])[1], tf.shape(feats[1])[2], 'BILINEAR')(feats[0]),
                     feats[1],
                     feats[2],
                     feats[3],
                     Resize(tf.shape(feats[3])[1], tf.shape(feats[3])[2], 'BILINEAR')(feats[4])]
        featmap_sizes = [tf.shape(featmap)[1:3] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)   # stride=4

        ins_pred_x_list, ins_pred_y_list, cate_pred_list = [], [], []
        for idx in range(len(self.seg_num_grids)):
            ins_feat = new_feats[idx]   # 给掩码分支
            cate_feat = new_feats[idx]  # 给分类分支

            # ============ ins branch (掩码分支，特征图形状是[N, mask_h, mask_w, grid]) ============
            ins_feat_x, ins_feat_y = layers.Lambda(concat_coord)(ins_feat)   # [N, h, w, c+1]、 [N, h, w, c+1]

            for ins_layer_x, ins_layer_y in zip(self.ins_convs_x, self.ins_convs_y):
                ins_feat_x = ins_layer_x(ins_feat_x)   # [N, h, w, 256]
                ins_feat_y = ins_layer_y(ins_feat_y)   # [N, h, w, 256]

            ins_feat_x = layers.UpSampling2D(2, interpolation='bilinear')(ins_feat_x)   # [N, 2*h, 2*w, 256]
            ins_feat_y = layers.UpSampling2D(2, interpolation='bilinear')(ins_feat_y)   # [N, 2*h, 2*w, 256]

            ins_pred_x = self.dsolo_ins_list_x[idx](ins_feat_x)   # [N, 2*h, 2*w, grid]，即[N, mask_h, mask_w, grid]
            ins_pred_y = self.dsolo_ins_list_y[idx](ins_feat_y)   # [N, 2*h, 2*w, grid]，即[N, mask_h, mask_w, grid]
            # 若输入图片大小为416x416，那么new_feats里图片大小应该为[52, 52, 26, 13, 13]，因为strides=[8, 8, 16, 32, 32]。
            # 那么对应的ins_pred_x大小应该为[104, 104, 52, 26, 26]；
            # 那么对应的ins_pred_y大小应该为[104, 104, 52, 26, 26]。

            # ============ cate branch (分类分支，特征图形状是[N, grid, grid, num_classes=80]) ============
            for i, cate_layer in enumerate(self.cate_convs):
                if i == self.cate_down_pos:   # 第0次都要插值成seg_num_grid x seg_num_grid的大小。
                    seg_num_grid = self.seg_num_grids[idx]
                    cate_feat = Resize(seg_num_grid, seg_num_grid, 'BILINEAR')(cate_feat)
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.dsolo_cate(cate_feat)   # 种类分支，通道数变成了80，[N, grid, grid, 80]

            # ============ 是否是预测状态 ============
            if eval:
                ins_pred_x = layers.Activation('sigmoid')(ins_pred_x)
                ins_pred_x = Resize(upsampled_size[0], upsampled_size[1], 'BILINEAR')(ins_pred_x)

                ins_pred_y = layers.Activation('sigmoid')(ins_pred_y)
                ins_pred_y = Resize(upsampled_size[0], upsampled_size[1], 'BILINEAR')(ins_pred_y)
                # 若输入图片大小为416x416，那么new_feats里图片大小应该为[52, 52, 26, 13, 13]，因为strides=[8, 8, 16, 32, 32]。
                # 那么此处的5个ins_pred_x大小应该为[104, 104, 104, 104, 104]；
                # 那么此处的5个ins_pred_y大小应该为[104, 104, 104, 104, 104]。即stride=4。训练时不会执行这里。

                cate_pred = layers.Activation('sigmoid')(cate_pred)
                # cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            ins_pred_x_list.append(ins_pred_x)
            ins_pred_y_list.append(ins_pred_y)
            cate_pred_list.append(cate_pred)
        return ins_pred_x_list+ ins_pred_y_list+ cate_pred_list
        # return ins_pred_x_list, ins_pred_y_list, cate_pred_list








