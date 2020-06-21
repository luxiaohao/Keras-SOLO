import mmcv

import numpy as np

dic2 = np.load('data.npz')
ins_preds_x = [dic2['ins_preds_x0'], dic2['ins_preds_x1'],
               dic2['ins_preds_x2'], dic2['ins_preds_x3'],
               dic2['ins_preds_x4']]
ins_preds_y = [dic2['ins_preds_y0'], dic2['ins_preds_y1'],
               dic2['ins_preds_y2'], dic2['ins_preds_y3'],
               dic2['ins_preds_y4']]
cate_preds = [dic2['cate_preds0'], dic2['cate_preds1'],
              dic2['cate_preds2'], dic2['cate_preds3'],
              dic2['cate_preds4']]


# ins_preds_x = [torch.Tensor(dic2['ins_preds_x0']), torch.Tensor(dic2['ins_preds_x1']),
#                torch.Tensor(dic2['ins_preds_x2']), torch.Tensor(dic2['ins_preds_x3']),
#                torch.Tensor(dic2['ins_preds_x4'])]
# ins_preds_y = [torch.Tensor(dic2['ins_preds_y0']), torch.Tensor(dic2['ins_preds_y1']),
#                torch.Tensor(dic2['ins_preds_y2']), torch.Tensor(dic2['ins_preds_y3']),
#                torch.Tensor(dic2['ins_preds_y4'])]
# cate_preds = [torch.Tensor(dic2['cate_preds0']), torch.Tensor(dic2['cate_preds1']),
#               torch.Tensor(dic2['cate_preds2']), torch.Tensor(dic2['cate_preds3']), torch.Tensor(dic2['cate_preds4'])]
gt_bbox_0 = dic2['gt_bbox_0']
gt_bbox_1 = dic2['gt_bbox_1']
gt_label_0 = dic2['gt_label_0']
gt_label_1 = dic2['gt_label_1']
gt_mask_0 = dic2['gt_mask_0']
gt_mask_1 = dic2['gt_mask_1']
gt_bbox_list = [gt_bbox_0, gt_bbox_1]
gt_label_list = [gt_label_0, gt_label_1]
gt_mask_list = [gt_mask_0, gt_mask_1]
featmap_sizes = [featmap.shape[-2:] for featmap in
                 ins_preds_x]


import cv2
from scipy import ndimage

class Gt2SoloTarget(object):
    """
    Generate SOLO targets.
    """

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
                 with_deform=False):
        super(Gt2SoloTarget, self).__init__()
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
        self.max_pos = 300

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[:2]
        an_hw = np.array(self.anchors) / np.array([[w, h]])

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_masks_raw,
                           featmap_sizes=None):
        # ins
        gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (   # 平均边长，几何平均数， [n, ]
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        gt_objs_per_layer = []
        gt_clss_per_layer = []
        gt_masks_per_layer = []
        gt_pos_idx_per_layer = []
        # 遍历每个输出层
        #           (1,     96)            8     [104, 104]     40
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):
            # [40, 40, 1]  objectness
            gt_objs = np.zeros([num_grid, num_grid, 1], dtype=np.float32)
            # [40, 40, 80]  种类one-hot
            gt_clss = np.zeros([num_grid, num_grid, self.num_classes], dtype=np.float32)
            # [?, 104, 104]  这一输出层的gt_masks，可能同一个掩码重复多次
            gt_masks = []
            # [self.max_pos, 3]    坐标以-2初始化
            # 前2个用于把正样本抽出来gather_nd()，后1个用于把掩码抽出来gather()。为了避免使用layers.where()后顺序没对上，所以不拆开写。
            gt_pos_idx = np.zeros([self.max_pos, 3], dtype=np.int32) - 2
            # 掩码计数
            p = 0

            # 这一张图片，所有物体，若平均边长在这个范围，这一输出层就负责预测。因为面积范围有交集，所以一个gt可以被分配到多个输出层上。
            hit_indices = np.where((gt_areas >= lower_bound) & (gt_areas <= upper_bound))[0]

            if len(hit_indices) == 0:   # 这一层没有正样本
                gt_objs_per_layer.append(gt_objs)   # 全是0
                gt_clss_per_layer.append(gt_clss)   # 全是0
                gt_masks = np.zeros([1, featmap_size[0], featmap_size[1]], dtype=np.uint8)   # 全是0，至少一张掩码，方便gather()
                gt_masks_per_layer.append(gt_masks)
                gt_pos_idx[0, :] = np.array([0, 0, 0], dtype=np.int32)   # 没有正样本，默认会抽第0行第0列格子，默认会抽这一层gt_mask里第0个掩码。
                gt_pos_idx_per_layer.append(gt_pos_idx)
                continue
            gt_bboxes_raw_this_layer = gt_bboxes_raw[hit_indices]   # shape=[m, 4]  这一层负责预测的物体的bbox
            gt_labels_raw_this_layer = gt_labels_raw[hit_indices]   # shape=[m, ]   这一层负责预测的物体的类别id
            gt_masks_raw_this_layer = gt_masks_raw[hit_indices]   # [m, ?, ?]

            half_ws = 0.5 * (gt_bboxes_raw_this_layer[:, 2] - gt_bboxes_raw_this_layer[:, 0]) * self.sigma   # shape=[m, ]  宽的一半
            half_hs = 0.5 * (gt_bboxes_raw_this_layer[:, 3] - gt_bboxes_raw_this_layer[:, 1]) * self.sigma   # shape=[m, ]  高的一半

            output_stride = stride / 2   # 因为网络最后对ins_feat_x、ins_feat_y进行上采样，所以stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks_raw_this_layer, gt_labels_raw_this_layer, half_hs, half_ws):
                if seg_mask.sum() < 10:   # 忽略太小的物体
                   continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)   # 也就是输入图片的大小
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)    # 求物体掩码的质心。scipy提供技术支持。
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))      # 物体质心落在了第几列格子
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))      # 物体质心落在了第几行格子

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))      # 物体左上角落在了第几行格子
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))    # 物体右下角落在了第几行格子
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))     # 物体左上角落在了第几列格子
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))   # 物体右下角落在了第几列格子

                # 物体的宽高并没有那么重要。将物体的左上角、右下角限制在质心所在的九宫格内。当物体很小时，物体的左上角、右下角、质心位于同一个格子。
                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                # down = top
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)
                # right = left

                # 40x40的网格，将负责预测gt的格子填上gt_objs和gt_clss，此处同YOLOv3
                # ins  [img_h, img_w]->[img_h/output_stride, img_w/output_stride]  将gt的掩码下采样output_stride倍。
                # seg_mask = cv2.resize(seg_mask, None, None, fx=1. / output_stride, fy=1. / output_stride, interpolation=cv2.INTER_LINEAR)
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        if gt_objs[i, j, 0] < 0.5:   # 这个格子没有被填过gt才可以填。
                            gt_objs[i, j, 0] = 1.0   # 此处同YOLOv3
                            gt_clss[i, j, gt_label] = 1.0   # 此处同YOLOv3
                            cp_mask = np.copy(seg_mask)
                            cp_mask = cp_mask[np.newaxis, :, :]
                            gt_masks.append(cp_mask)
                            gt_pos_idx[p, :] = np.array([i, j, p], dtype=np.int32)   # 前2个用于把正样本抽出来gather_nd()，后1个用于把掩码抽出来gather()。
                            p += 1
            gt_masks = np.concatenate(gt_masks, axis=0)

            gt_objs_per_layer.append(gt_objs)
            gt_clss_per_layer.append(gt_clss)
            gt_masks_per_layer.append(gt_masks)
            gt_pos_idx_per_layer.append(gt_pos_idx)
        return gt_objs_per_layer, gt_clss_per_layer, gt_masks_per_layer, gt_pos_idx_per_layer


gt2SoloTarget = Gt2SoloTarget()

batch_gt_objs, batch_gt_clss, batch_gt_masks, batch_gt_pos_idx = [], [], [], []
batch_size = 2
for i in range(batch_size):
    gt_objs_per_layer, gt_clss_per_layer, gt_masks_per_layer, gt_pos_idx_per_layer = gt2SoloTarget.solo_target_single(gt_bbox_list[i], gt_label_list[i], gt_mask_list[i], featmap_sizes)
    batch_gt_objs.append(gt_objs_per_layer)
    batch_gt_clss.append(gt_clss_per_layer)
    batch_gt_masks.append(gt_masks_per_layer)
    batch_gt_pos_idx.append(gt_pos_idx_per_layer)
    print()




import keras
import keras.layers as layers
import keras.backend as K
import tensorflow as tf





batch_size = 2
num_layers = 5



batch_ins_preds_x = []
batch_ins_preds_y = []
batch_cate_preds = []
for lid in range(num_layers):
    batch_layer_ins_preds_x = layers.Input(name='batch_layer%d_ins_preds_x' % (lid, ), shape=(None, None, None), dtype='float32')
    batch_layer_ins_preds_y = layers.Input(name='batch_layer%d_ins_preds_y' % (lid, ), shape=(None, None, None), dtype='float32')
    batch_layer_cate_preds = layers.Input(name='batch_layer%d_cate_preds' % (lid, ), shape=(None, None, None), dtype='float32')
    batch_ins_preds_x.append(batch_layer_ins_preds_x)
    batch_ins_preds_y.append(batch_layer_ins_preds_y)
    batch_cate_preds.append(batch_layer_cate_preds)



batch_gt_objs_tensors = []
batch_gt_clss_tensors = []
batch_gt_masks_tensors = []
batch_gt_pos_idx_tensors = []
for bid in range(batch_size):
    sample_gt_objs = []
    sample_gt_clss = []
    sample_gt_masks = []
    sample_gt_pos_idx = []
    for lid in range(num_layers):
        sample_layer_gt_objs = layers.Input(name='sample%d_layer%d_gt_objs' % (bid, lid), batch_shape=(None, None, 1), dtype='float32')
        sample_layer_gt_clss = layers.Input(name='sample%d_layer%d_gt_clss' % (bid, lid), batch_shape=(None, None, None), dtype='float32')
        sample_layer_gt_masks = layers.Input(name='sample%d_layer%d_gt_masks' % (bid, lid), batch_shape=(None, None, None), dtype='float32')
        sample_layer_gt_pos_idx = layers.Input(name='sample%d_layer%d_gt_pos_idx' % (bid, lid), batch_shape=(None, 3), dtype='int32')
        sample_gt_objs.append(sample_layer_gt_objs)
        sample_gt_clss.append(sample_layer_gt_clss)
        sample_gt_masks.append(sample_layer_gt_masks)
        sample_gt_pos_idx.append(sample_layer_gt_pos_idx)
    batch_gt_objs_tensors.append(sample_gt_objs)
    batch_gt_clss_tensors.append(sample_gt_clss)
    batch_gt_masks_tensors.append(sample_gt_masks)
    batch_gt_pos_idx_tensors.append(sample_gt_pos_idx)




def dice_loss(pred_mask, gt_mask, gt_obj):
    a = tf.reduce_sum(pred_mask * gt_mask, axis=[1, 2])
    b = tf.reduce_sum(pred_mask * pred_mask, axis=[1, 2]) + 0.001
    c = tf.reduce_sum(gt_mask * gt_mask, axis=[1, 2]) + 0.001
    d = (2 * a) / (b + c)
    loss_mask_mask = tf.reshape(gt_obj, (-1, ))   # 掩码损失的掩码。
    return (1-d) * loss_mask_mask
    # return a


def loss_layer():
    num_ins = 0.  # 记录这一批图片的正样本个数
    loss_clss, loss_masks = [], []
    for bid in range(batch_size):
        for lid in range(num_layers):
            # ================ 掩码损失 ======================
            pred_mask_x = batch_ins_preds_x[lid][bid]
            pred_mask_y = batch_ins_preds_y[lid][bid]
            pred_mask_x = tf.transpose(pred_mask_x, perm=[2, 0, 1])
            pred_mask_y = tf.transpose(pred_mask_y, perm=[2, 0, 1])

            gt_objs = batch_gt_objs_tensors[bid][lid]
            gt_masks = batch_gt_masks_tensors[bid][lid]
            pmidx = batch_gt_pos_idx_tensors[bid][lid]

            idx_sum = tf.reduce_sum(pmidx, axis=1)
            keep = tf.where(idx_sum > -1)
            keep = tf.reshape(keep, (-1, ))
            pmidx = tf.gather(pmidx, keep)

            yx_idx = pmidx[:, :2]
            y_idx = pmidx[:, 0]
            x_idx = pmidx[:, 1]
            m_idx = pmidx[:, 2]

            # 抽出来
            gt_obj = tf.gather_nd(gt_objs, yx_idx)
            mask_y = tf.gather(pred_mask_y, y_idx)
            mask_x = tf.gather(pred_mask_x, x_idx)
            gt_mask = tf.gather(gt_masks, m_idx)

            # 正样本数量
            num_ins += tf.reduce_sum(gt_obj)

            pred_mask = tf.sigmoid(mask_x) * tf.sigmoid(mask_y)
            loss_mask = dice_loss(pred_mask, gt_mask, gt_obj)
            loss_masks.append(loss_mask)


            # ================ 分类损失 ======================
            gamma = 2.0
            alpha = 0.25
            pred_conf = batch_cate_preds[lid][bid]
            pred_conf = tf.sigmoid(pred_conf)
            gt_clss = batch_gt_clss_tensors[bid][lid]
            pos_loss = gt_clss * (0 - tf.log(pred_conf + 1e-9)) * tf.pow(1 - pred_conf, gamma) * alpha
            neg_loss = (1 - gt_clss) * (0 - tf.log(1 - pred_conf + 1e-9)) * tf.pow(pred_conf, gamma) * (1 - alpha)
            clss_loss = pos_loss + neg_loss
            clss_loss = tf.reduce_sum(clss_loss, axis=[0, 1])
            loss_clss.append(clss_loss)
    loss_masks = tf.concat(loss_masks, axis=0)
    ins_loss_weight = 3.0
    loss_masks = tf.reduce_sum(loss_masks) * ins_loss_weight
    loss_masks = loss_masks / (num_ins + 1e-9)   # 损失同原版SOLO，之所以不直接用tf.reduce_mean()，是因为多了一些0损失占位，分母并不等于num_ins。

    loss_clss = tf.concat(loss_clss, axis=0)
    clss_loss_weight = 1.0
    loss_clss = tf.reduce_sum(loss_clss) * clss_loss_weight
    loss_clss = loss_clss / (num_ins + 1e-9)

    loss = loss_masks + loss_clss
    return loss

aa = loss_layer()


feed_dict = {}
for bid in range(batch_size):
    for lid in range(num_layers):
        feed_dict[batch_gt_objs_tensors[bid][lid]] = batch_gt_objs[bid][lid]
        feed_dict[batch_gt_clss_tensors[bid][lid]] = batch_gt_clss[bid][lid]
        feed_dict[batch_gt_masks_tensors[bid][lid]] = batch_gt_masks[bid][lid]
        feed_dict[batch_gt_pos_idx_tensors[bid][lid]] = batch_gt_pos_idx[bid][lid]


for lid in range(num_layers):
    feed_dict[batch_ins_preds_x[lid]] = ins_preds_x[lid].transpose(0, 2, 3, 1)
    feed_dict[batch_ins_preds_y[lid]] = ins_preds_y[lid].transpose(0, 2, 3, 1)
    feed_dict[batch_cate_preds[lid]] = cate_preds[lid].transpose(0, 2, 3, 1)

sess = K.get_session()

aaa00 = sess.run(aa, feed_dict=feed_dict)


print()
# 不同图片的gt掩码汇合
# ins
ins_labels = [np.concatenate([ins_labels_level_img[ins_ind_labels_level_img, ...]
                         for ins_labels_level_img, ins_ind_labels_level_img in
                         zip(ins_labels_level, ins_ind_labels_level)], 0)
              for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]


ins_labels2 = []
# 遍历每个输出层，遍历5次
for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list)):
    this_layer = []
    # 遍历每张图片，遍历batch_size次
    for ins_labels_level_img, ins_ind_labels_level_img in zip(ins_labels_level, ins_ind_labels_level):
        one_img_mask = ins_labels_level_img[ins_ind_labels_level_img, ...]
        this_layer.append(one_img_mask)
    this_layer = np.concatenate(this_layer, 0)   # 不同图片的这个输出层的gt掩码 拼接
    ins_labels2.append(this_layer)












ins_preds_x_final = [np.concatenate([ins_preds_level_img_x[ins_ind_labels_level_img[:, 1], ...]
                                for ins_preds_level_img_x, ins_ind_labels_level_img in
                                zip(ins_preds_level_x, ins_ind_labels_level)], 0)
                     for ins_preds_level_x, ins_ind_labels_level in
                     zip(ins_preds_x, zip(*ins_ind_label_list_xy))]


# 把被负责预测gt的格子的ins_preds_x抽出来。yolov3是用掩码忽略，solo是只抽出正样本的张量。
ins_preds_x_final2 = []
# 遍历每个输出张量ins_preds_x，遍历5次
for ins_preds_level_x, ins_ind_labels_level in zip(ins_preds_x, zip(*ins_ind_label_list_xy)):
    t1 = []
    # 遍历每张图片，遍历batch_size次
    for ins_preds_level_img_x, ins_ind_labels_level_img in zip(ins_preds_level_x, ins_ind_labels_level):
        # 这张图片的这个输出层的ins_preds_x输出
        one_img_mask = ins_preds_level_img_x[ins_ind_labels_level_img[:, 1], ...]
        t1.append(one_img_mask)
    t1 = np.concatenate(t1, 0)
    ins_preds_x_final2.append(t1)



# ins_preds_y_final = [np.concatenate([ins_preds_level_img_y[ins_ind_labels_level_img[:, 0], ...]
#                                 for ins_preds_level_img_y, ins_ind_labels_level_img in
#                                 zip(ins_preds_level_y, ins_ind_labels_level)], 0)
#                      for ins_preds_level_y, ins_ind_labels_level in
#                      zip(ins_preds_y, zip(*ins_ind_label_list_xy))]

num_ins = 0.

print()





