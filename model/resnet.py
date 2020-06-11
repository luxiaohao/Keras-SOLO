#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import keras
import tensorflow as tf
import keras.layers as layers
from keras import backend as K
from keras.engine.topology import Layer



def _3x3conv(x, filters2, use_dcn):
    if use_dcn:
        pass
    else:
        x = layers.Conv2D(filters2, 3,
                   padding='same',
                   strides=1,
                   use_bias=False,
                   activation='linear',
                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.advanced_activations.ReLU()(x)
    return x

def conv_block(input_tensor, filters, use_dcn=False, stride=2):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, 1,
               padding='valid',
               strides=stride,
               use_bias=False,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.advanced_activations.ReLU()(x)

    x = _3x3conv(x, filters2, use_dcn)

    x = layers.Conv2D(filters3, 1,
               padding='valid',
               strides=1,
               use_bias=False,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    x = layers.BatchNormalization()(x)


    shortcut = layers.Conv2D(filters3, 1,
               padding='valid',
               strides=stride,
               use_bias=False,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.advanced_activations.ReLU()(x)
    return x


def identity_block(input_tensor, filters, use_dcn=False):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, 1,
               padding='valid',
               strides=1,
               use_bias=False,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.advanced_activations.ReLU()(x)

    x = _3x3conv(x, filters2, use_dcn)

    x = layers.Conv2D(filters3, 1,
               padding='valid',
               strides=1,
               use_bias=False,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.advanced_activations.ReLU()(x)
    return x

def stage1(x):
    x = layers.Conv2D(64, 7,
               padding='same',
               strides=2,
               use_bias=False,
               activation='linear',
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.advanced_activations.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    return x


def Resnet50(inputs, use_dcn):
    # stage1
    x = stage1(inputs)
    # stage2
    x = conv_block(x, [64, 64, 256], stride=1)
    x = identity_block(x, [64, 64, 256])
    s4 = identity_block(x, [64, 64, 256])
    # stage3
    x = conv_block(s4, [128, 128, 512], use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], use_dcn=use_dcn)
    s8 = identity_block(x, [128, 128, 512], use_dcn=use_dcn)
    # stage4
    x = conv_block(s8, [256, 256, 1024], use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    x = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    s16 = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    # stage5
    x = conv_block(s16, [512, 512, 2048], use_dcn=use_dcn)
    x = identity_block(x, [512, 512, 2048], use_dcn=use_dcn)
    s32 = identity_block(x, [512, 512, 2048], use_dcn=use_dcn)

    return [s4, s8, s16, s32]


def Resnet101(inputs, use_dcn):
    # stage1
    x = stage1(inputs)
    # stage2
    x = conv_block(x, [64, 64, 256], stride=1)
    x = identity_block(x, [64, 64, 256])
    s4 = identity_block(x, [64, 64, 256])
    # stage3
    x = conv_block(s4, [128, 128, 512], use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], use_dcn=use_dcn)
    x = identity_block(x, [128, 128, 512], use_dcn=use_dcn)
    s8 = identity_block(x, [128, 128, 512], use_dcn=use_dcn)
    # stage4
    x = conv_block(s8, [256, 256, 1024], use_dcn=use_dcn)
    for i in range(1, 22):
        x = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    s16 = identity_block(x, [256, 256, 1024], use_dcn=use_dcn)
    # stage5
    x = conv_block(s16, [512, 512, 2048], use_dcn=use_dcn)
    x = identity_block(x, [512, 512, 2048], use_dcn=use_dcn)
    s32 = identity_block(x, [512, 512, 2048], use_dcn=use_dcn)

    return [s4, s8, s16, s32]