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
import keras
import keras.layers as layers

from model.solo import SOLO

num_classes = 80


filename = 'images/test/000000000019.jpg'

import cv2
import mmcv

results = {}
img = cv2.imread(filename)

results['filename'] = filename
results['img'] = img
results['img_shape'] = img.shape
results['ori_shape'] = img.shape


# Resize
input_shape = (1333, 800)   # (w, h)
results['scale'] = input_shape   # (w, h)
results['flip'] = False
img = results['img']
ori_h, ori_w = img.shape[:2]
scale_x = float(input_shape[0]) / ori_w
scale_y = float(input_shape[1]) / ori_h
scale_factor = min(scale_x, scale_y)
img = cv2.resize(img, None, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
results['img'] = img
results['img_shape'] = img.shape
results['pad_shape'] = img.shape  # in case that there is no padding
results['scale_factor'] = scale_factor
results['keep_ratio'] = True




# Normalize
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
mean = np.array(mean, dtype=np.float32)
std = np.array(std, dtype=np.float32)

img = results['img']
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = (img - mean) / std
results['img'] = img


# _pad_img.填充，让宽高可以被32整除
size = None
size_divisor = 32
pad_val = 0
img = results['img']
if size is not None:
    padded_img = mmcv.impad(results['img'], size)
elif size_divisor is not None:
    padded_img = mmcv.impad_to_multiple(
        results['img'], size_divisor, pad_val=pad_val)
results['img'] = padded_img
results['pad_shape'] = padded_img.shape
results['pad_fixed_size'] = size
results['pad_size_divisor'] = size_divisor



inputs = layers.Input(shape=(None, None, 3))
# inputs = layers.Input(shape=(416, 416, 3))

eval=True
outs = SOLO(inputs, use_dcn=False, eval=eval)
model = keras.models.Model(inputs=inputs, outputs=outs)
# model.summary()
# keras.utils.vis_utils.plot_model(model, to_file='solo.png', show_shapes=True)

model.load_weights('solo.h5', by_name=True)

pimage = np.expand_dims(padded_img, axis=0)


aa = model.predict(pimage)

p = 0


x0 = aa[p][0].transpose(2, 0, 1)
p += 1
x1 = aa[p][0].transpose(2, 0, 1)
p += 1
x2 = aa[p][0].transpose(2, 0, 1)
p += 1
x3 = aa[p][0].transpose(2, 0, 1)
p += 1
x4 = aa[p][0].transpose(2, 0, 1)
p += 1
seg_pred_list_x = np.concatenate([x0, x1, x2, x3, x4], axis=0)


y0 = aa[p][0].transpose(2, 0, 1)
p += 1
y1 = aa[p][0].transpose(2, 0, 1)
p += 1
y2 = aa[p][0].transpose(2, 0, 1)
p += 1
y3 = aa[p][0].transpose(2, 0, 1)
p += 1
y4 = aa[p][0].transpose(2, 0, 1)
p += 1
seg_pred_list_y = np.concatenate([y0, y1, y2, y3, y4], axis=0)


c0 = np.reshape(aa[p][0], (-1, 80))
p += 1
c1 = np.reshape(aa[p][0], (-1, 80))
p += 1
c2 = np.reshape(aa[p][0], (-1, 80))
p += 1
c3 = np.reshape(aa[p][0], (-1, 80))
p += 1
c4 = np.reshape(aa[p][0], (-1, 80))
p += 1
cate_pred_list = np.concatenate([c0, c1, c2, c3, c4], axis=0)



dic = {}
dic['seg_pred_list_x'] = seg_pred_list_x
dic['seg_pred_list_y'] = seg_pred_list_y
dic['cate_pred_list'] = cate_pred_list
np.savez('data', **dic)



print()



def forward_single(x, idx, eval, upsampled_size):
    ins_feat = x     # [N, c, h, w]
    cate_feat = x    # [N, c, h, w]
    # ins branch
    # concat coord

    seg_num_grids = [40, 36, 24, 16, 12]
    trans_size2 = torch.Tensor(seg_num_grids).pow(2)
    trans_size = torch.Tensor(seg_num_grids).pow(2).cumsum(0).long()



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




