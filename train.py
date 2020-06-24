#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import cv2
from collections import deque
import math
import json
import time
import shutil
import threading
import datetime
import keras
import random
import copy
import numpy as np
import keras.layers as layers
import os
import tensorflow as tf
from keras import backend as K

from config import DecoupledSOLO_R50_FPN_Config
from model.head import DecoupledSOLOHead
from model.neck import FPN
from model.resnet import Resnet
from model.solo import SOLO
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


# 显存分配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


def dice_loss(pred_mask, gt_mask, gt_obj):
    a = tf.reduce_sum(pred_mask * gt_mask, axis=[1, 2])
    b = tf.reduce_sum(pred_mask * pred_mask, axis=[1, 2]) + 0.001
    c = tf.reduce_sum(gt_mask * gt_mask, axis=[1, 2]) + 0.001
    d = (2 * a) / (b + c)
    loss_mask_mask = tf.reshape(gt_obj, (-1, ))   # 掩码损失的掩码。
    return (1-d) * loss_mask_mask


def solo_loss2(args, batch_size, num_layers):
    p = 0

    ins_pred_x_list = []
    for i in range(num_layers):
        ins_pred_x_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    ins_pred_y_list = []
    for i in range(num_layers):
        ins_pred_y_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    cate_pred_list = []
    for i in range(num_layers):
        cate_pred_list.append(args[p])    # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    batch_gt_objs_tensors = []
    for i in range(num_layers):
        batch_gt_objs_tensors.append(args[p])
        p += 1

    batch_gt_clss_tensors = []
    for i in range(num_layers):
        batch_gt_clss_tensors.append(args[p])
        p += 1

    batch_gt_masks_tensors = []
    for i in range(num_layers):
        batch_gt_masks_tensors.append(args[p])
        p += 1

    batch_gt_pos_idx_tensors = []
    for i in range(num_layers):
        batch_gt_pos_idx_tensors.append(args[p])
        p += 1


    # ================= 计算损失 =================
    num_ins = 0.  # 记录这一批图片的正样本个数
    loss_clss, loss_masks = [], []
    for bid in range(batch_size):
        for lid in range(num_layers):
            # ================ 掩码损失 ======================
            pred_mask_x = ins_pred_x_list[lid][bid]
            pred_mask_y = ins_pred_y_list[lid][bid]
            pred_mask_x = tf.transpose(pred_mask_x, perm=[2, 0, 1])
            pred_mask_y = tf.transpose(pred_mask_y, perm=[2, 0, 1])

            gt_objs = batch_gt_objs_tensors[lid][bid]
            gt_masks = batch_gt_masks_tensors[lid][bid]
            pmidx = batch_gt_pos_idx_tensors[lid][bid]

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
            pred_conf = cate_pred_list[lid][bid]
            pred_conf = tf.sigmoid(pred_conf)
            gt_clss = batch_gt_clss_tensors[lid][bid]
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

    return [loss_masks, loss_clss]

def solo_loss(args, batch_size, num_layers):
    p = 0

    ins_pred_x_list = []
    for i in range(num_layers):
        ins_pred_x_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    ins_pred_y_list = []
    for i in range(num_layers):
        ins_pred_y_list.append(args[p])   # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    cate_pred_list = []
    for i in range(num_layers):
        cate_pred_list.append(args[p])    # 从小感受野 到 大感受野 （从多格子 到 少格子）
        p += 1

    batch_gt_objs_tensors = []
    for i in range(num_layers):
        batch_gt_objs_tensors.append(args[p])
        p += 1

    batch_gt_clss_tensors = []
    for i in range(num_layers):
        batch_gt_clss_tensors.append(args[p])
        p += 1

    batch_gt_masks_tensors = []
    for i in range(num_layers):
        batch_gt_masks_tensors.append(args[p])
        p += 1

    batch_gt_pos_idx_tensors = []
    for i in range(num_layers):
        batch_gt_pos_idx_tensors.append(args[p])
        p += 1


    # ================= 计算损失 =================
    num_ins = 0.  # 记录这一批图片的正样本个数
    loss_clss, loss_masks = [], []
    for bid in range(batch_size):
        for lid in range(num_layers):
            # ================ 掩码损失 ======================
            pred_mask_x = ins_pred_x_list[lid][bid]
            pred_mask_y = ins_pred_y_list[lid][bid]
            pred_mask_x = tf.transpose(pred_mask_x, perm=[2, 0, 1])
            pred_mask_y = tf.transpose(pred_mask_y, perm=[2, 0, 1])

            gt_objs = batch_gt_objs_tensors[lid][bid]
            gt_masks = batch_gt_masks_tensors[lid][bid]
            pmidx = batch_gt_pos_idx_tensors[lid][bid]

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
            pred_conf = cate_pred_list[lid][bid]
            pred_conf = tf.sigmoid(pred_conf)
            gt_clss = batch_gt_clss_tensors[lid][bid]
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

    return [loss_masks, loss_clss]
    # return loss_masks


def multi_thread_op(i, samples, decodeImage, context, train_dataset, with_mixup, mixupImage,
                     photometricDistort, randomCrop, randomFlipImage, padBox):
    samples[i] = decodeImage(samples[i], context, train_dataset)
    if with_mixup:
        samples[i] = mixupImage(samples[i], context)
    samples[i] = photometricDistort(samples[i], context)
    samples[i] = randomCrop(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = padBox(samples[i], context)

if __name__ == '__main__':
    cfg = DecoupledSOLO_R50_FPN_Config()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    batch_size = cfg.batch_size
    num_layers = 5

    # 步id，无需设置，会自动读。
    iter_id = 0

    # 多尺度训练
    inputs = layers.Input(shape=(None, None, 3))
    resnet = Resnet(50)
    fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
    head = DecoupledSOLOHead()
    solo = SOLO(resnet, fpn, head)
    outs = solo(inputs, cfg, eval=False)
    model_body = keras.models.Model(inputs=inputs, outputs=outs)


    # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'solo.h5'、'./weights/step00001000.h5'这些。）
    pattern = cfg.pattern
    if pattern == 1:
        model_body.load_weights(cfg.model_path, by_name=True, skip_mismatch=True)
        strs = cfg.model_path.split('step')
        if len(strs) == 2:
            iter_id = int(strs[1][:8])

        # 冻结，使得需要的显存减少。6G的卡建议这样配置。11G的卡建议不冻结。
        # freeze_before = 'conv2d_60'
        freeze_before = 'conv2d_70'
        for i in range(len(model_body.layers)):
            ly = model_body.layers[i]
            if ly.name == freeze_before:
                break
            else:
                ly.trainable = False
    elif pattern == 0:
        pass

    # 标记张量
    batch_gt_objs_tensors = []
    batch_gt_clss_tensors = []
    batch_gt_masks_tensors = []
    batch_gt_pos_idx_tensors = []
    for lid in range(num_layers):
        sample_layer_gt_objs = layers.Input(name='layer%d_gt_objs' % (lid, ), shape=(None, None, 1), dtype='float32')
        sample_layer_gt_clss = layers.Input(name='layer%d_gt_clss' % (lid, ), shape=(None, None, num_classes), dtype='float32')
        sample_layer_gt_masks = layers.Input(name='layer%d_gt_masks' % (lid, ), shape=(None, None, None), dtype='float32')
        sample_layer_gt_pos_idx = layers.Input(name='layer%d_gt_pos_idx' % (lid, ), shape=(None, 3), dtype='int32')
        batch_gt_objs_tensors.append(sample_layer_gt_objs)
        batch_gt_clss_tensors.append(sample_layer_gt_clss)
        batch_gt_masks_tensors.append(sample_layer_gt_masks)
        batch_gt_pos_idx_tensors.append(sample_layer_gt_pos_idx)

    loss_list = layers.Lambda(solo_loss, name='solo_loss',
                           arguments={'batch_size': batch_size, 'num_layers': num_layers
                                      })([*model_body.output, *batch_gt_objs_tensors, *batch_gt_clss_tensors, *batch_gt_masks_tensors, *batch_gt_pos_idx_tensors])
    model = keras.models.Model([model_body.input, *batch_gt_objs_tensors, *batch_gt_clss_tensors, *batch_gt_masks_tensors, *batch_gt_pos_idx_tensors], loss_list)
    model.summary()
    # keras.utils.vis_utils.plot_model(model_body, to_file='solo.png', show_shapes=True)

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            val_images = dataset['images']

    with_mixup = cfg.with_mixup
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(with_mixup=with_mixup)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage()                   # mixup增强
    photometricDistort = PhotometricDistort()   # 颜色扭曲
    randomCrop = RandomCrop()                   # 随机裁剪
    randomFlipImage = RandomFlipImage()         # 随机翻转
    # 增加PadBox()处理也是为了防止RandomShape()出现Process finished with exit code -1073740940 (0xC0000374)
    padBox = PadBox(cfg.num_max_boxes)          # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。

    # batch_transforms
    # 6个分辨率(w, h)，随机选一个分辨率(w, h)训练。也随机选一种插值方式。原版SOLO中，因为设定了size_divisor=32，
    # 所以被填充黑边的宽（或者高）会填充最少的黑边使得被32整除。所以一个batch最后所有的图片的大小有很大概率是不同的，
    # pytorch版为了用一个张量(bz, c, h2, w2)表示这一批不同分辨率的图片，所有图片会向最大分辨率的图片看齐（通过填充黑边0）。
    # 而且h2, w2很大概率只有一个等于被选中的h, w，另一个是填充的最小的能被32整除的。
    # 这里和原作稍有不同，按照size_divisor=None处理，即统一填充到被选中的分辨率(w, h)。在考虑后面改为跟随原作。
    # randomShape = RandomShape(sizes=[(320, 320), (320, 320)])
    randomShape = RandomShape()
    normalizeImage = NormalizeImage(is_scale=False, is_channel_first=False)  # 图片归一化。
    gt2SoloTarget = Gt2SoloTarget()

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    model.compile(loss={'solo_loss': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(lr=cfg.lr))

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    best_ap_list = [0.0, 0]  #[map, iter]
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        # np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.max_iters - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op, args=(i, samples, decodeImage, context, train_dataset, with_mixup, mixupImage,
                                                                   photometricDistort, randomCrop, randomFlipImage, padBox))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # debug  看数据增强后的图片。由于有随机裁剪，所以有的物体掩码不完整。
            # if os.path.exists('temp/'): shutil.rmtree('temp/')
            # os.mkdir('temp/')
            # samples = randomShape(samples, context)
            # for r, sample in enumerate(samples):
            #     img = sample['image']
            #     gt_score = sample['gt_score']
            #     gt_mask = sample['gt_mask']
            #     aa = gt_mask.transpose(2, 0, 1)
            #     cv2.imwrite('temp/%d.jpg'%r, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            #     for rr, sc in enumerate(gt_score):
            #         if sc > 0:
            #             m = gt_mask[:, :, rr]
            #             cv2.imwrite('temp/%d_%d.jpg'%(r, rr), m*255)

            # batch_transforms
            # 是randomShape导致了Process finished with exit code -1073740940 (0xC0000374)
            samples = randomShape(samples, context)
            samples = normalizeImage(samples, context)
            batch_image, batch_gt_objs, batch_gt_clss, batch_gt_masks, batch_gt_pos_idx = gt2SoloTarget(samples, context)

            batch_xs = [batch_image, *batch_gt_objs, *batch_gt_clss, *batch_gt_masks, *batch_gt_pos_idx]
            y_true = [np.zeros(batch_size), np.zeros(batch_size)]
            # y_true = [np.zeros((batch_size, )), np.zeros((batch_size, ))]
            losses = model.train_on_batch(batch_xs, y_true)

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, mask_loss: {:.6f}, clss_loss: {:.6f}, eta: {}'.format(
                    iter_id, losses[0], losses[1], losses[2], eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.save_iter == 0:
                save_path = './weights/step%.8d.h5' % iter_id
                model.save(save_path)
                path_dir = os.listdir('./weights')
                steps = []
                names = []
                for name in path_dir:
                    if name[len(name) - 2:len(name)] == 'h5' and name[0:4] == 'step':
                        step = int(name[4:12])
                        steps.append(step)
                        names.append(name)
                if len(steps) > 10:
                    i = steps.index(min(steps))
                    os.remove('./weights/'+names[i])
                logger.info('Save model to {}'.format(save_path))

            # ==================== eval ====================
            # if iter_id % cfg.eval_iter == 0:
            #     box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_batch_size, _clsid2catid, cfg.draw_image)
            #     logger.info("box ap: %.3f" % (box_ap[0], ))
            #
            #     # 以box_ap作为标准
            #     ap = box_ap
            #     if ap[0] > best_ap_list[0]:
            #         best_ap_list[0] = ap[0]
            #         best_ap_list[1] = iter_id
            #         model.save('./weights/best_model.h5')
            #     logger.info("Best test ap: {}, in iter: {}".format(
            #         best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.max_iters:
                logger.info('Done.')
                exit(0)

