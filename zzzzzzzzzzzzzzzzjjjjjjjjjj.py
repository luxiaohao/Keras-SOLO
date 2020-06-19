import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv, roi_align
from mmdet.core import multi_apply, bbox2roi, matrix_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule

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

@HEADS.register_module
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
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.ins_convs_x = nn.ModuleList()
        self.ins_convs_y = nn.ModuleList()
        self.cate_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels + 1 if i == 0 else self.seg_feat_channels
            self.ins_convs_x.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            self.ins_convs_y.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.dsolo_ins_list_x = nn.ModuleList()
        self.dsolo_ins_list_y = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.dsolo_ins_list_x.append(
                nn.Conv2d(
                    self.seg_feat_channels, seg_num_grid, 3, padding=1))
            self.dsolo_ins_list_y.append(
                nn.Conv2d(
                    self.seg_feat_channels, seg_num_grid, 3, padding=1))
        self.dsolo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.ins_convs_x:
            normal_init(m.conv, std=0.01)
        for m in self.ins_convs_y:
            normal_init(m.conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_ins = bias_init_with_prob(0.01)
        for m in self.dsolo_ins_list_x:
            normal_init(m, std=0.01, bias=bias_ins)
        for m in self.dsolo_ins_list_y:
            normal_init(m, std=0.01, bias=bias_ins)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.dsolo_cate, std=0.01, bias=bias_cate)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        ins_pred_x, ins_pred_y, cate_pred = multi_apply(self.forward_single, new_feats,
                                                        list(range(len(self.seg_num_grids))),
                                                        eval=eval, upsampled_size=upsampled_size)
        return ins_pred_x, ins_pred_y, cate_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
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

        for ins_layer_x, ins_layer_y in zip(self.ins_convs_x, self.ins_convs_y):
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
        return ins_pred_x, ins_pred_y, cate_pred

    def loss(self,
             ins_preds_x,
             ins_preds_y,
             cate_preds,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in
                         ins_preds_x]   # [[104, 104], [104, 104], [52, 52], [26, 26], [26, 26]]




        # ins_label_list       里有形状为[?1, 104, 104]、[?2, 104, 104]、[?3, 52, 52]、...的5个数组。是不支持拼接的。其他list同理。
        # cate_label_list      里有形状为[40, 40]、[36, 36]、[24, 24]、...的5个数组。
        # ins_ind_label_list   里有形状为[?1, ]、[?2, ]、[?3, ]、...的5个数组。全是1。
        # ins_ind_label_list_xy里有形状为[?1, 2]、[?2, 2]、[?3, 2]、...的5个数组。[?1, 2]表示正样本在cate_label_list[0]里的下标。
        ins_label_list, cate_label_list, ins_ind_label_list, ins_ind_label_list_xy = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)

        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        ins_preds_x_final = [torch.cat([ins_preds_level_img_x[ins_ind_labels_level_img[:, 1], ...]
                                for ins_preds_level_img_x, ins_ind_labels_level_img in
                                zip(ins_preds_level_x, ins_ind_labels_level)], 0)
                     for ins_preds_level_x, ins_ind_labels_level in
                     zip(ins_preds_x, zip(*ins_ind_label_list_xy))]

        ins_preds_y_final = [torch.cat([ins_preds_level_img_y[ins_ind_labels_level_img[:, 0], ...]
                                  for ins_preds_level_img_y, ins_ind_labels_level_img in
                                  zip(ins_preds_level_y, ins_ind_labels_level)], 0)
                       for ins_preds_level_y, ins_ind_labels_level in
                       zip(ins_preds_y, zip(*ins_ind_label_list_xy))]

        num_ins = 0.
        # dice loss
        loss_ins = []
        for input_x, input_y, target in zip(ins_preds_x_final, ins_preds_y_final, ins_labels):
            mask_n = input_x.size(0)
            if mask_n == 0:
                continue
            num_ins += mask_n
            input = (input_x.sigmoid())*(input_y.sigmoid())
            loss_ins.append(dice_loss(input, target))

        loss_ins = torch.cat(loss_ins).mean() * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate)

    def solo_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               featmap_sizes=None):
        '''

        Args: gt_xxx都是这一批里一张图片的标记
            gt_bboxes_raw:    shape=[n, 4]
            gt_labels_raw:    shape=[n, ]
            gt_masks_raw:     shape=[n, ?, ?]
            featmap_sizes:    [[104, 104], [104, 104], [52, 52], [26, 26], [26, 26]]

        Returns:

        '''

        device = gt_labels_raw[0].device
        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (   # 面积， [n, ]
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        ins_ind_label_list_xy = []

        # 遍历每个输出层
        #           (1,     96)            8     [104, 104]     40
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):

            # [40*40, 104, 104]  这一输出层，一个格子占用一张掩码图。
            ins_label = torch.zeros([num_grid**2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            # [40, 40]  种类id
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            # [40*40, ]  ins_label中，某个格子是否被gt占用
            ins_ind_label = torch.zeros([num_grid**2], dtype=torch.bool, device=device)

            # 这一张图片，所有物体，若面积在这个范围，这一输出层就负责预测。因为面积范围有交集，所以一个gt可以被分配到多个输出层上。
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()

            if len(hit_indices) == 0:
                ins_label = torch.zeros([1, featmap_size[0], featmap_size[1]], dtype=torch.uint8,
                                        device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label = torch.zeros([1], dtype=torch.bool, device=device)
                ins_ind_label_list.append(ins_ind_label)
                ins_ind_label_list_xy.append(cate_label.nonzero())
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]   # shape=[m, 4]  这一层负责预测的物体的bbox
            gt_labels = gt_labels_raw[hit_indices]   # shape=[m, xxxx]  这一层负责预测的物体的类别id
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]   # [m, ?, ?]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma   # shape=[m, ]  宽的一半
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma   # shape=[m, ]  高的一半

            output_stride = stride / 2   # 因为网络最后对ins_feat_x、ins_feat_y进行上采样，所以stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
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
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # squared。 # [40, 40]的网格，将物体占的格子填上类别id。
                cate_label[top:(down+1), left:(right+1)] = gt_label
                # ins  [img_h, img_w]->[img_h/output_stride, img_w/output_stride]  将gt的掩码下采样output_stride倍。
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True   # ins_label中，这个位置的格子被gt占用

            ins_label = ins_label[ins_ind_label]   # ins_label中，抽出被gt占用的格子的掩码，减少内存占用。
            ins_label_list.append(ins_label)     # [?, 104, 104]，  ?>=m，因为一个gt可能占用多个格子。

            cate_label_list.append(cate_label)   # [40, 40]的网格，类别id。

            ins_ind_label = ins_ind_label[ins_ind_label]  # 全是1
            ins_ind_label_list.append(ins_ind_label)  #  [?, ]

            ins_ind_label_list_xy.append(cate_label.nonzero())   # 非零元素的下标。[[gt1_coord1, gt1_coord2], [gt2_coord1, gt2_coord2], ...]

        # ins_label_list       里有形状为[?1, 104, 104]、[?2, 104, 104]、[?3, 52, 52]、...的5个数组。是不支持拼接的。其他list同理。
        # cate_label_list      里有形状为[40, 40]、[36, 36]、[24, 24]、...的5个数组。
        # ins_ind_label_list   里有形状为[?1, ]、[?2, ]、[?3, ]、...的5个数组。全是1。
        # ins_ind_label_list_xy里有形状为[?1, 2]、[?2, 2]、[?3, 2]、...的5个数组。[?1, 2]表示正样本在cate_label_list[0]里的下标。
        return ins_label_list, cate_label_list, ins_ind_label_list, ins_ind_label_list_xy

    def get_seg(self, seg_preds_x, seg_preds_y, cate_preds, img_metas, cfg, rescale=None):
        assert len(seg_preds_x) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds_x[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)   # [40, 40, 80]、[36, 36, 80]、...
            ]
            seg_pred_list_x = [
                seg_preds_x[i][img_id].detach() for i in range(num_levels)   # [40, 104, 104]、[36, 104, 104]、...
            ]
            seg_pred_list_y = [
                seg_preds_y[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']         # [1333, 800]
            scale_factor = img_metas[img_id]['scale_factor']   # [sc_x, sc_y]
            ori_shape = img_metas[img_id]['ori_shape']         # [img_h, img_w]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)    # 同一张图片5个输出层的输出汇合  [40*40+36*36+24*24+16*16+12*12, 80]
            seg_pred_list_x = torch.cat(seg_pred_list_x, dim=0)  # 同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            seg_pred_list_y = torch.cat(seg_pred_list_y, dim=0)  # 同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]

            result = self.get_seg_single(cate_pred_list, seg_pred_list_x, seg_pred_list_y,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

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
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]

        inds = inds.nonzero()
        trans_diff = torch.index_select(trans_diff, dim=0, index=inds[:, 0])
        seg_diff = torch.index_select(seg_diff, dim=0, index=inds[:, 0])
        num_grids = torch.index_select(num_grids, dim=0, index=inds[:, 0])
        strides = torch.index_select(strides, dim=0, index=inds[:, 0])

        y_inds = (inds[:, 0] - trans_diff) // num_grids
        x_inds = (inds[:, 0] - trans_diff) % num_grids
        y_inds += seg_diff
        x_inds += seg_diff

        cate_labels = inds[:, 1]
        seg_masks_soft = seg_preds_x[x_inds, ...] * seg_preds_y[y_inds, ...]
        seg_masks = seg_masks_soft > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()
        keep = sum_masks > strides

        seg_masks_soft = seg_masks_soft[keep, ...]
        seg_masks = seg_masks[keep, ...]
        cate_scores = cate_scores[keep]
        sum_masks = sum_masks[keep]
        cate_labels = cate_labels[keep]
        # mask scoring
        seg_score = (seg_masks_soft * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_score

        if len(cate_scores) == 0:
            return None

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        seg_masks = seg_masks[sort_inds, :, :]
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
