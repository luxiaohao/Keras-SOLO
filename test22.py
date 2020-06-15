#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================

# import keras
# import tensorflow as tf
# import keras.layers as layers
# from keras import backend as K
# from keras.engine.topology import Layer

import torch
import numpy as np


num_classes = 80






def forward_single(x, idx, eval, upsampled_size):
    ins_feat = x     # [N, c, h, w]
    cate_feat = x    # [N, c, h, w]
    # ins branch
    # concat coord
    x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)   # [w, ]
    y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)   # [h, ]
    y, x = torch.meshgrid(y_range, x_range)   # y [h, w]     x [h, w]
    y = y.expand([ins_feat.shape[0], 1, -1, -1])   # [N, 1, h, w]
    x = x.expand([ins_feat.shape[0], 1, -1, -1])   # [N, 1, h, w]
    ins_feat_x = torch.cat([ins_feat, x], 1)    # [N, c+1, h, w]
    ins_feat_y = torch.cat([ins_feat, y], 1)    # [N, c+1, h, w]

    '''for ins_layer_x, ins_layer_y in zip(self.ins_convs_x, self.ins_convs_y):
        ins_feat_x = ins_layer_x(ins_feat_x)
        ins_feat_y = ins_layer_y(ins_feat_y)

    ins_feat_x = F.interpolate(ins_feat_x, scale_factor=2, mode='bilinear')
    ins_feat_y = F.interpolate(ins_feat_y, scale_factor=2, mode='bilinear')

    ins_pred_x = self.dsolo_ins_list_x[idx](ins_feat_x)
    ins_pred_y = self.dsolo_ins_list_y[idx](ins_feat_y)

    # cate branch
    for i, cate_layer in enumerate(self.cate_convs):
        if i == self.cate_down_pos:
            seg_num_grid = self.seg_num_grids[idx]
            cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
        cate_feat = cate_layer(cate_feat)

    cate_pred = self.dsolo_cate(cate_feat)

    if eval:
        ins_pred_x = F.interpolate(ins_pred_x.sigmoid(), size=upsampled_size, mode='bilinear')
        ins_pred_y = F.interpolate(ins_pred_y.sigmoid(), size=upsampled_size, mode='bilinear')
        cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
    return ins_pred_x, ins_pred_y, cate_pred'''
    return 0



a = np.zeros((8, 256, 6, 4), np.float32)

x = torch.from_numpy(a)


upsampled_size = (8, 8)  # stride=4

aa = forward_single(x, 0, True, upsampled_size)





# inputs = layers.Input(shape=(None, None, 3))
# inputs = layers.Input(shape=(416, 416, 3))
# outs = SOLO(inputs, num_classes, use_dcn=False)
# model = keras.models.Model(inputs=inputs, outputs=outs)
# model.summary()
# keras.utils.vis_utils.plot_model(model, to_file='solo.png', show_shapes=True)




