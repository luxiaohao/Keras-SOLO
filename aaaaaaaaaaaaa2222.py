import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

import keras
import keras.layers as layers
import keras.backend as K
import tensorflow as tf

INF = 1e8

from scipy import ndimage

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d




def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)   True、False组成的掩码
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor):  shape (n, )      n个物体的面积

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = tf.shape(cate_labels)[0]   # 物体数
    seg_masks = tf.reshape(tf.cast(seg_masks, tf.float32), (n_samples, -1))   # [n, h*w]  掩码由True、False变成0、1
    # inter.
    inter_matrix = tf.matmul(seg_masks, seg_masks, transpose_b=True)   # [n, n] 自己乘以自己的转置。两两之间的交集面积。
    # union.
    sum_masks_x = tf.tile(sum_masks[tf.newaxis, :], [n_samples, 1])     # [n, n]  sum_masks重复了n行得到sum_masks_x
    # iou.
    iou_matrix = inter_matrix / (sum_masks_x + tf.transpose(sum_masks_x, [1, 0]) - inter_matrix)
    rows = tf.range(0, n_samples, 1, 'int32')
    cols = tf.range(0, n_samples, 1, 'int32')
    rows = tf.tile(tf.reshape(rows, (1, -1)), [n_samples, 1])
    cols = tf.tile(tf.reshape(cols, (-1, 1)), [1, n_samples])
    tri_mask = tf.cast(rows > cols, 'float32')
    iou_matrix = tri_mask * iou_matrix   # [n, n]   只取上三角部分

    # label_specific matrix.
    cate_labels_x = tf.tile(cate_labels[tf.newaxis, :], [n_samples, 1])     # [n, n]  cate_labels重复了n行得到cate_labels_x
    label_matrix = tf.cast(tf.equal(cate_labels_x, tf.transpose(cate_labels_x, [1, 0])), tf.float32)
    label_matrix = tri_mask * label_matrix   # [n, n]   只取上三角部分

    # IoU compensation
    compensate_iou = tf.reduce_max(iou_matrix * label_matrix, axis=0)
    compensate_iou = tf.tile(compensate_iou[tf.newaxis, :], [n_samples, 1])     # [n, n]
    compensate_iou = tf.transpose(compensate_iou, [1, 0])      # [n, n]

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # # matrix nms
    if kernel == 'gaussian':
        decay_matrix = tf.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = tf.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = tf.reduce_min((decay_matrix / compensate_matrix), axis=0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = tf.reduce_min(decay_matrix, axis=0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update




class DecoupledSOLOHead(nn.Module):
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
        # self.loss_cate = build_loss(loss_cate)
        # self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg



    def get_seg_single(self,
                       cate_preds,
                       seg_preds_x,
                       seg_preds_y,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):
        '''
        Args:
            cate_preds:    同一张图片5个输出层的输出汇合  [40*40+36*36+24*24+16*16+12*12, 80]
            seg_preds_x:   同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            seg_preds_y:   同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            featmap_size:  [s4, s4]        一维张量  1-D Tensor
            img_shape:     [800, 1216, 3]  一维张量  1-D Tensor
            ori_shape:     [427, 640, 3]   一维张量  1-D Tensor
            scale_factor:  800/427
            cfg:
            rescale:
            debug:

        Returns:

        '''
        # trans trans_diff.
        seg_num_grids = tf.zeros([len(self.seg_num_grids), ], tf.int32) + np.array(self.seg_num_grids)
        trans_size = tf.cumsum(tf.pow(seg_num_grids, 2))
        seg_size = tf.cumsum(seg_num_grids)    # [40, 40+36, 40+36+24, ...]

        trans_diff = []
        seg_diff = []
        num_grids = []
        strides = []
        n_stage = len(self.seg_num_grids)   # 5个输出层
        for ind_ in range(n_stage):
            if ind_ == 0:
                # 第0个输出层的分类分支在cate_preds中的偏移是0
                trans_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32)
                # 第0个输出层的掩码分支在seg_preds_x中的偏移是0
                seg_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32)
            else:
                # 第1个输出层的分类分支在cate_preds中的偏移是40*40，第2个输出层的分类分支在cate_preds中的偏移是40*40+36*36，...
                trans_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32) + trans_size[ind_ - 1]
                # 第0个输出层的掩码分支在seg_preds_x中的偏移是40，第0个输出层的掩码分支在seg_preds_x中的偏移是40+36，...
                seg_diff_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32) + seg_size[ind_ - 1]
            # 第0个输出层的一行（或一列）的num_grids是40，第1个输出层的一行（或一列）的num_grids是36，...
            num_grids_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.int32) + self.seg_num_grids[ind_]
            # 第0个输出层的stride是8，第1个输出层的stride是8，...
            strides_ = tf.zeros([self.seg_num_grids[ind_] ** 2, ], tf.float32) + float(self.strides[ind_])

            trans_diff.append(trans_diff_)
            seg_diff.append(seg_diff_)
            num_grids.append(num_grids_)
            strides.append(strides_)
        trans_diff = tf.concat(trans_diff, axis=0)   # [3872, ]
        seg_diff = tf.concat(seg_diff, axis=0)       # [3872, ]
        num_grids = tf.concat(num_grids, axis=0)     # [3872, ]
        strides = tf.concat(strides, axis=0)         # [3872, ]

        # process. 处理。
        inds = tf.where(cate_preds > cfg.score_thr)   # [[3623, 17], [3623, 60], [3639, 17], ...]   分数超过阈值的物体所在格子
        cate_scores = tf.gather_nd(cate_preds, inds)

        trans_diff = tf.gather(trans_diff, inds[:, 0])   # [3472, 3472, 3472, ...]   格子所在输出层的分类分支在cate_preds中的偏移
        seg_diff = tf.gather(seg_diff, inds[:, 0])       # [100, 100, 100, ...]      格子所在输出层的掩码分支在seg_preds_x中的偏移
        num_grids = tf.gather(num_grids, inds[:, 0])     # [16, 16, 16, ...]         格子所在输出层每一行有多少个格子
        strides = tf.gather(strides, inds[:, 0])         # [32, 32, 32, ...]         格子所在输出层的stride

        loc = tf.cast(inds[:, 0], tf.int32)
        y_inds = (loc - trans_diff) // num_grids   # 格子行号
        x_inds = (loc - trans_diff) % num_grids    # 格子列号
        y_inds += seg_diff   # 格子行号在seg_preds_y中的绝对位置
        x_inds += seg_diff   # 格子列号在seg_preds_x中的绝对位置

        cate_labels = inds[:, 1]   # 类别
        mask_x = tf.gather(seg_preds_x, x_inds)   # [11, s4, s4]
        mask_y = tf.gather(seg_preds_y, y_inds)   # [11, s4, s4]
        seg_masks_soft = mask_x * mask_y    # [11, s4, s4]  物体的mask，逐元素相乘得到。
        seg_masks = seg_masks_soft > cfg.mask_thr
        sum_masks = tf.reduce_sum(tf.cast(seg_masks, tf.float32), axis=[1, 2])   # [11, ]  11个物体的面积
        keep = tf.where(sum_masks > strides)   # 面积大于这一层的stride才保留

        seg_masks_soft = tf.gather_nd(seg_masks_soft, keep)   # 用概率表示的掩码
        seg_masks = tf.gather_nd(seg_masks, keep)             # 用True、False表示的掩码
        cate_scores = tf.gather_nd(cate_scores, keep)    # 类别得分
        sum_masks = tf.gather_nd(sum_masks, keep)        # 面积
        cate_labels = tf.gather_nd(cate_labels, keep)    # 类别
        # mask scoring   是1的像素的 概率总和 占 面积（是1的像素数） 的比重
        seg_score = tf.reduce_sum(seg_masks_soft * tf.cast(seg_masks, tf.float32), axis=[1, 2]) / sum_masks
        cate_scores *= seg_score   # 类别得分乘上这个比重得到新的类别得分。因为有了mask scoring机制，所以分数一般比其它算法如yolact少。

        # I hate tensorflow.
        def exist_objs_1(cate_scores, seg_masks_soft, seg_masks, sum_masks, cate_labels):
            # sort and keep top nms_pre
            k = tf.shape(cate_scores)[0]
            _, sort_inds = tf.nn.top_k(cate_scores, k=k, sorted=True)   # [7, 5, 8, ...] 降序。最大值的下标，第2大值的下标，...
            sort_inds = sort_inds[:cfg.nms_pre]   # 最多cfg.nms_pre个。
            seg_masks_soft = tf.gather(seg_masks_soft, sort_inds)   # 按照分数降序
            seg_masks = tf.gather(seg_masks, sort_inds)             # 按照分数降序
            cate_scores = tf.gather(cate_scores, sort_inds)
            sum_masks = tf.gather(sum_masks, sort_inds)
            cate_labels = tf.gather(cate_labels, sort_inds)

            # Matrix NMS
            cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                     kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

            keep = tf.where(cate_scores > cfg.update_thr)   # 大于cfg.update_thr才保留
            keep = tf.reshape(keep, (-1, ))
            seg_masks_soft = tf.gather(seg_masks_soft, keep)
            cate_scores = tf.gather(cate_scores, keep)
            cate_labels = tf.gather(cate_labels, keep)

            # I hate tensorflow.
            def exist_objs_2(cate_scores, seg_masks_soft, cate_labels):
                # sort and keep top_k
                k = tf.shape(cate_scores)[0]
                _, sort_inds = tf.nn.top_k(cate_scores, k=k, sorted=True)
                sort_inds = sort_inds[:cfg.max_per_img]
                seg_masks_soft = tf.gather(seg_masks_soft, sort_inds)
                cate_scores = tf.gather(cate_scores, sort_inds)
                cate_labels = tf.gather(cate_labels, sort_inds)

                # 插值前处理
                seg_masks_soft = tf.transpose(seg_masks_soft, [1, 2, 0])
                seg_masks_soft = seg_masks_soft[tf.newaxis, :, :, :]

                seg_masks_soft = tf.image.resize_images(seg_masks_soft, [featmap_size[0] * 4, featmap_size[1] * 4], method=tf.image.ResizeMethod.BILINEAR)
                seg_masks = tf.image.resize_images(seg_masks_soft, [ori_shape[0], ori_shape[1]], method=tf.image.ResizeMethod.BILINEAR)

                # 插值后处理
                seg_masks = tf.reshape(seg_masks, tf.shape(seg_masks)[1:])
                seg_masks = tf.transpose(seg_masks, [2, 0, 1])

                seg_masks = tf.cast(seg_masks > cfg.mask_thr, tf.float32)
                return seg_masks, cate_labels, cate_scores

            # I hate tensorflow.
            def no_objs_2():
                seg_masks = tf.zeros([1, ], tf.float32) - 1.0
                cate_labels = tf.zeros([1, ], tf.int64) - 1
                cate_scores = tf.zeros([1, ], tf.float32) - 1.0
                return seg_masks, cate_labels, cate_scores

            # 是否有物体
            # I hate tensorflow.
            seg_masks, cate_labels, cate_scores = tf.cond(tf.equal(tf.shape(cate_scores)[0], 0),
                                                          no_objs_2,
                                                          lambda: exist_objs_2(cate_scores, seg_masks_soft, cate_labels))
            return seg_masks, cate_labels, cate_scores

        # I hate tensorflow.
        def no_objs_1():
            seg_masks = tf.zeros([1, ], tf.float32) - 1.0
            cate_labels = tf.zeros([1, ], tf.int64) - 1
            cate_scores = tf.zeros([1, ], tf.float32) - 1.0
            return seg_masks, cate_labels, cate_scores

        # 是否有物体
        # I hate tensorflow.
        seg_masks, cate_labels, cate_scores = tf.cond(tf.equal(tf.shape(cate_scores)[0], 0),
                                                      no_objs_1,
                                                      lambda: exist_objs_1(cate_scores, seg_masks_soft, seg_masks, sum_masks, cate_labels))
        return seg_masks, cate_labels, cate_scores


import numpy as np

dic2 = np.load('data.npz')
seg_pred_list_x2 = dic2['seg_pred_list_x']
seg_pred_list_y2 = dic2['seg_pred_list_y']
cate_pred_list2 = dic2['cate_pred_list']


seg_pred_list_x = layers.Input(name='seg_pred_list_x', batch_shape=(None, None, None))
seg_pred_list_y = layers.Input(name='seg_pred_list_y', batch_shape=(None, None, None))
cate_pred_list = layers.Input(name='cate_pred_list', batch_shape=(None, None))




featmap_size2 = [200, 304]
img_shape2 = [800, 1216, 3]
ori_shape2 = [427, 640, 3]
scale_factor = 800/427

featmap_size2 = np.array(featmap_size2)
img_shape2 = np.array(img_shape2)
ori_shape2 = np.array(ori_shape2)


featmap_size = layers.Input(name='featmap_size', batch_shape=(None, ))
img_shape = layers.Input(name='img_shape', batch_shape=(None, ))
ori_shape = layers.Input(name='ori_shape', batch_shape=(None, ))



cfg = None
rescale = False


class TrainConfig(object):
    """
    train.py里需要的配置
    """
    def __init__(self):
        self.nms_pre = 500
        self.score_thr = 0.1
        # self.mask_thr = 0.5
        self.mask_thr = 0.005
        self.update_thr = 0.001
        self.kernel = 'gaussian'
        self.sigma = 2.0
        self.max_per_img = 100

cfg = TrainConfig()

aaa = DecoupledSOLOHead(81, 256,
        stacked_convs=7,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        with_deform=False)
aa = aaa.get_seg_single(cate_pred_list, seg_pred_list_x, seg_pred_list_y,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)


sess = K.get_session()
aaa00 = sess.run(aa, feed_dict={seg_pred_list_x: seg_pred_list_x2, seg_pred_list_y: seg_pred_list_y2, cate_pred_list: cate_pred_list2,
                                featmap_size: featmap_size2, img_shape: img_shape2, ori_shape: ori_shape2,})

masks = aaa00[0]
classes = aaa00[1]
scores = aaa00[2]

import cv2

for yy, m in enumerate(masks):
    print(scores[yy])
    print(classes[yy])
    cv2.imwrite('%d.png'%yy, m*255)
    print()


# aaa01 = np.mean(aaa00)
# aaa02 = np.var(aaa00)

print()





