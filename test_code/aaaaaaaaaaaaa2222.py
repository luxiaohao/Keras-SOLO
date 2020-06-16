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
    n_samples = len(cate_labels)   # 物体数
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()   # [n, h*w]  掩码由True、False变成0、1
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))   # [n, n] 自己乘以自己的转置。两两之间的交集面积。
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)     # [n, n]  sum_masks重复了n行得到sum_masks_x
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)   # 只取上三角部分
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)   # [n, n]  cate_labels重复了n行得到cate_labels_x
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
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

    def get_seg_single2(self,
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
            featmap_size:  [s4, s4]
            img_shape:     [1333, 800]
            ori_shape:     [img_h, img_w]
            scale_factor:  [sc_x, sc_y]
            cfg:
            rescale:
            debug:

        Returns:

        '''


        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)   # (s4*4, s4*4)

        # trans trans_diff.
        trans_size = torch.Tensor(self.seg_num_grids).pow(2).cumsum(0).long()   # [40*40, 40*40+36*36, 40*40+36*36+24*24, ...]
        trans_diff = torch.ones(trans_size[-1].item(), device=cate_preds.device).long()   # [3872, ] 每个格子填一个1
        num_grids = torch.ones(trans_size[-1].item(), device=cate_preds.device).long()    # [3872, ] 每个格子填一个1

        seg_size = torch.Tensor(self.seg_num_grids).cumsum(0).long()    # [40, 40+36, 40+36+24, ...]
        seg_diff = torch.ones(trans_size[-1].item(), device=cate_preds.device).long()     # [3872, ] 每个格子填一个1
        strides = torch.ones(trans_size[-1].item(), device=cate_preds.device)             # [3872, ] 每个格子填一个1

        n_stage = len(self.seg_num_grids)   # 5个输出层
        trans_diff[:trans_size[0]] *= 0     # 第0个输出层的分类分支在cate_preds中的偏移是0
        seg_diff[:trans_size[0]] *= 0       # 第0个输出层的掩码分支在seg_preds_x中的偏移是0
        num_grids[:trans_size[0]] *= self.seg_num_grids[0]   # 第0个输出层的一行（或一列）的num_grids是40
        strides[:trans_size[0]] *= self.strides[0]           # 第0个输出层的stride是8

        for ind_ in range(1, n_stage):
            # 第1个输出层的分类分支在cate_preds中的偏移是40*40，第2个输出层的分类分支在cate_preds中的偏移是40*40+36*36，...
            trans_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= trans_size[ind_ - 1]
            # 第0个输出层的掩码分支在seg_preds_x中的偏移是40，第0个输出层的掩码分支在seg_preds_x中的偏移是40+36，...
            seg_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= seg_size[ind_ - 1]

            num_grids[trans_size[ind_ - 1]:trans_size[ind_]] *= self.seg_num_grids[ind_]   # 第1个输出层的一行（或一列）的num_grids是36，...
            strides[trans_size[ind_ - 1]:trans_size[ind_]] *= self.strides[ind_]           # 第1个输出层的stride是8，...

        # process.
        inds = (cate_preds > cfg.score_thr)   # [[3623, 17], [3623, 60], [3639, 17], ...]   分数超过阈值的物体所在格子
        cate_scores = cate_preds[inds]

        inds = inds.nonzero()
        trans_diff = torch.index_select(trans_diff, dim=0, index=inds[:, 0])   # [3472, 3472, 3472, ...]   格子所在输出层的分类分支在cate_preds中的偏移
        seg_diff = torch.index_select(seg_diff, dim=0, index=inds[:, 0])       # [100, 100, 100, ...]      格子所在输出层的掩码分支在seg_preds_x中的偏移
        num_grids = torch.index_select(num_grids, dim=0, index=inds[:, 0])     # [16, 16, 16, ...]         格子所在输出层每一行有多少个格子
        strides = torch.index_select(strides, dim=0, index=inds[:, 0])         # [32, 32, 32, ...]         格子所在输出层的stride

        y_inds = (inds[:, 0] - trans_diff) // num_grids   # 格子行号
        x_inds = (inds[:, 0] - trans_diff) % num_grids    # 格子列号
        y_inds += seg_diff   # 格子行号在seg_preds_y中的绝对位置
        x_inds += seg_diff   # 格子列号在seg_preds_x中的绝对位置

        cate_labels = inds[:, 1]   # 类别
        mask_x = seg_preds_x[x_inds, ...]   # [11, s4, s4]
        mask_y = seg_preds_y[y_inds, ...]   # [11, s4, s4]
        seg_masks_soft = mask_x * mask_y    # [11, s4, s4]  物体的mask，逐元素相乘得到。
        seg_masks = seg_masks_soft > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()   # [11, ]  11个物体的面积
        keep = sum_masks > strides   # 面积大于这一层的stride才保留

        seg_masks_soft = seg_masks_soft[keep, ...]   # 用概率表示的掩码
        seg_masks = seg_masks[keep, ...]             # 用0、1表示的掩码
        cate_scores = cate_scores[keep]    # 类别得分
        sum_masks = sum_masks[keep]        # 面积
        cate_labels = cate_labels[keep]    # 类别
        # mask scoring   是1的像素的 概率总和 占 面积（是1的像素数） 的比重
        seg_score = (seg_masks_soft * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_score   # 类别得分乘上这个比重得到新的类别得分。因为有了mask scoring机制，所以分数一般比其它算法如yolact少。

        if len(cate_scores) == 0:   # 没有物体，直接return
            return None

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)   # [7, 5, 8, ...] 降序。最大值的下标，第2大值的下标，...
        if len(sort_inds) > cfg.nms_pre:    # 最多cfg.nms_pre个。
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]   # 按照分数降序
        seg_masks = seg_masks[sort_inds, :, :]             # 按照分数降序
        cate_scores = cate_scores[sort_inds]
        sum_masks = sum_masks[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        keep = cate_scores >= cfg.update_thr
        seg_masks_soft = seg_masks_soft[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_masks_soft = F.interpolate(seg_masks_soft.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_masks_soft,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores



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
            featmap_size:  [s4, s4]
            img_shape:     [800, 1216, 3]
            ori_shape:     [427, 640]
            scale_factor:  800/427
            cfg:
            rescale:
            debug:

        Returns:

        '''


        # overall info.
        # h = tf.shape(img_shape)[0]
        # w = tf.shape(img_shape)[1]
        # upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)   # (s4*4, s4*4)

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

        # process.
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

        # if len(cate_scores) == 0:   # 没有物体，直接return
        #     return None

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

        # keep = cate_scores >= cfg.update_thr
        # seg_masks_soft = seg_masks_soft[keep, :, :]
        # cate_scores = cate_scores[keep]
        # cate_labels = cate_labels[keep]

        return cate_scores


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
ori_shape2 = [427, 640]
scale_factor = 800/427

featmap_size2 = np.array(featmap_size2)
featmap_size2 = np.array(img_shape2)
featmap_size2 = np.array(ori_shape2)


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
        self.update_thr = 0.05
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

# aaa01 = np.mean(aaa00)
# aaa02 = np.var(aaa00)

print()





