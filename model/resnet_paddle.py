#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import paddle.fluid as fluid
import paddle.fluid.layers as P
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay





def _3x3conv(x, filters2, pre_name, is_test, trainable, use_dcn):
    if use_dcn:
        offset_mask = P.conv2d(
            input=x,
            num_filters=27,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv2.conv_offset_mask.weight", trainable=trainable),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=pre_name + ".conv2.conv_offset_mask.bias", trainable=trainable))
        offset = offset_mask[:, :18, :, :]
        mask = offset_mask[:, 18:, :, :]
        mask = P.sigmoid(mask)
        x = P.deformable_conv(input=x, offset=offset, mask=mask,
            num_filters=filters2,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            deformable_groups=1,
            im2col_step=1,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv2.weight", trainable=trainable),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=pre_name + ".conv2.bias", trainable=trainable))
    else:
        x = P.conv2d(
            input=x,
            num_filters=filters2,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv2.weight", trainable=trainable),
            bias_attr=False)
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn2.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn2.bias'),
        moving_mean_name=pre_name + '.bn2.running_mean',
        moving_variance_name=pre_name + '.bn2.running_var')
    x = P.relu(x)
    return x


def conv_block(input_tensor, filters, pre_name, is_test, trainable, use_dcn=False, stride=2):
    filters1, filters2, filters3 = filters

    x = P.conv2d(
        input=input_tensor,
        num_filters=filters1,
        filter_size=1,
        stride=stride,
        padding=0,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv1.weight", trainable=trainable),
        bias_attr=False)
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn1.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn1.bias'),
        moving_mean_name=pre_name + '.bn1.running_mean',
        moving_variance_name=pre_name + '.bn1.running_var')
    x = P.relu(x)

    x = _3x3conv(x, filters2, pre_name, is_test, trainable, use_dcn)

    x = P.conv2d(
        input=x,
        num_filters=filters3,
        filter_size=1,
        stride=1,
        padding=0,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv3.weight", trainable=trainable),
        bias_attr=False)
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn3.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn3.bias'),
        moving_mean_name=pre_name + '.bn3.running_mean',
        moving_variance_name=pre_name + '.bn3.running_var')


    shortcut = P.conv2d(
        input=input_tensor,
        num_filters=filters3,
        filter_size=1,
        stride=stride,
        padding=0,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".downsample.0.weight", trainable=trainable),
        bias_attr=False)
    shortcut = P.batch_norm(
        input=shortcut,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.downsample.1.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.downsample.1.bias'),
        moving_mean_name=pre_name + '.downsample.1.running_mean',
        moving_variance_name=pre_name + '.downsample.1.running_var')

    x = P.elementwise_add(x=x, y=shortcut, act='relu')
    return x


def identity_block(input_tensor, filters, pre_name, is_test, trainable, use_dcn=False):
    filters1, filters2, filters3 = filters

    x = P.conv2d(
        input=input_tensor,
        num_filters=filters1,
        filter_size=1,
        stride=1,
        padding=0,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv1.weight", trainable=trainable),
        bias_attr=False)
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn1.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn1.bias'),
        moving_mean_name=pre_name + '.bn1.running_mean',
        moving_variance_name=pre_name + '.bn1.running_var')
    x = P.relu(x)

    x = _3x3conv(x, filters2, pre_name, is_test, trainable, use_dcn)

    x = P.conv2d(
        input=x,
        num_filters=filters3,
        filter_size=1,
        stride=1,
        padding=0,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=pre_name + ".conv3.weight", trainable=trainable),
        bias_attr=False)
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn3.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=pre_name + '.bn3.bias'),
        moving_mean_name=pre_name + '.bn3.running_mean',
        moving_variance_name=pre_name + '.bn3.running_var')

    x = P.elementwise_add(x=x, y=input_tensor, act='relu')
    return x

def stage1(x, is_test, trainable):
    x = P.conv2d(x, 64, filter_size=7, stride=2, padding=3,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="backbone.conv1.weight", trainable=trainable),
                 bias_attr=False)
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name='backbone.bn1.weight'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name='backbone.bn1.bias'),
        moving_mean_name='backbone.bn1.running_mean',
        moving_variance_name='backbone.bn1.running_var')
    x = P.relu(x)
    x = P.pool2d(x, pool_size=3, pool_type="max", pool_stride=2, pool_padding=1)
    return x


def Resnet50(inputs, is_test, trainable, use_dcn):
    # stage1
    x = stage1(inputs, is_test, trainable)
    # stage2
    x = conv_block(x, [64, 64, 256], 'backbone.layers.0.0', is_test, trainable, stride=1)
    x = identity_block(x, [64, 64, 256], 'backbone.layers.0.1', is_test, trainable)
    x = identity_block(x, [64, 64, 256], 'backbone.layers.0.2', is_test, trainable)
    # stage3
    x = conv_block(x, [128, 128, 512], 'backbone.layers.1.0', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], 'backbone.layers.1.1', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], 'backbone.layers.1.2', is_test, trainable, use_dcn=use_dcn)
    s8 = identity_block(x, [128, 128, 512], 'backbone.layers.1.3', is_test, trainable, use_dcn=use_dcn)
    # stage4
    x = conv_block(s8, [256, 256, 1024], 'backbone.layers.2.0', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], 'backbone.layers.2.1', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], 'backbone.layers.2.2', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], 'backbone.layers.2.3', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], 'backbone.layers.2.4', is_test, trainable, use_dcn=use_dcn)
    s16 = identity_block(x, [256, 256, 1024], 'backbone.layers.2.5', is_test, trainable, use_dcn=use_dcn)
    # stage5
    x = conv_block(s16, [512, 512, 2048], 'backbone.layers.3.0', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [512, 512, 2048], 'backbone.layers.3.1', is_test, trainable, use_dcn=use_dcn)
    s32 = identity_block(x, [512, 512, 2048], 'backbone.layers.3.2', is_test, trainable, use_dcn=use_dcn)

    return s8, s16, s32


def Resnet101(inputs, is_test, trainable, use_dcn):
    # stage1
    x = stage1(inputs, is_test, trainable)
    # stage2
    x = conv_block(x, [64, 64, 256], 'backbone.layers.0.0', is_test, trainable, stride=1)
    x = identity_block(x, [64, 64, 256], 'backbone.layers.0.1', is_test, trainable)
    x = identity_block(x, [64, 64, 256], 'backbone.layers.0.2', is_test, trainable)
    # stage3
    x = conv_block(x, [128, 128, 512], 'backbone.layers.1.0', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], 'backbone.layers.1.1', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], 'backbone.layers.1.2', is_test, trainable, use_dcn=use_dcn)
    s8 = identity_block(x, [128, 128, 512], 'backbone.layers.1.3', is_test, trainable, use_dcn=use_dcn)
    # stage4
    x = conv_block(s8, [256, 256, 1024], 'backbone.layers.2.0', is_test, trainable, use_dcn=use_dcn)
    for i in range(1, 22):
        x = identity_block(x, [256, 256, 1024], 'backbone.layers.2.%d'%i, is_test, trainable, use_dcn=use_dcn)
    s16 = identity_block(x, [256, 256, 1024], 'backbone.layers.2.22', is_test, trainable, use_dcn=use_dcn)
    # stage5
    x = conv_block(s16, [512, 512, 2048], 'backbone.layers.3.0', is_test, trainable, use_dcn=use_dcn)
    x = identity_block(x, [512, 512, 2048], 'backbone.layers.3.1', is_test, trainable, use_dcn=use_dcn)
    s32 = identity_block(x, [512, 512, 2048], 'backbone.layers.3.2', is_test, trainable, use_dcn=use_dcn)

    return s8, s16, s32