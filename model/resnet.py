#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
import keras.layers as layers
from model.custom_layers import Conv2dUnit, Conv3x3


class ConvBlock(object):
    def __init__(self, filters, use_dcn=False, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, strides=stride, padding='valid', use_bias=False, bn=1, act='relu')
        self.conv2 = Conv3x3(filters2, use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, strides=1, padding='valid', use_bias=False, bn=1, act=None)

        self.conv4 = Conv2dUnit(filters3, 1, strides=stride, padding='valid', use_bias=False, bn=1, act=None)
        self.act = layers.advanced_activations.ReLU()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = layers.add([x, shortcut])
        x = self.act(x)
        return x


class IdentityBlock(object):
    def __init__(self, filters, use_dcn=False):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, strides=1, padding='valid', use_bias=False, bn=1, act='relu')
        self.conv2 = Conv3x3(filters2, use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, strides=1, padding='valid', use_bias=False, bn=1, act=None)

        self.act = layers.advanced_activations.ReLU()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.add([x, input_tensor])
        x = self.act(x)
        return x

class Resnet(object):
    def __init__(self, depth, use_dcn=False):
        super(Resnet, self).__init__()
        assert depth in [50, 101]
        self.conv1 = Conv2dUnit(64, 7, strides=2, padding='same', use_bias=False, bn=1, act='relu')
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        # stage2
        self.stage2_0 = ConvBlock([64, 64, 256], stride=1)
        self.stage2_1 = IdentityBlock([64, 64, 256])
        self.stage2_2 = IdentityBlock([64, 64, 256])

        # stage3
        self.stage3_0 = ConvBlock([128, 128, 512], use_dcn=use_dcn)
        self.stage3_1 = IdentityBlock([128, 128, 512], use_dcn=use_dcn)
        self.stage3_2 = IdentityBlock([128, 128, 512], use_dcn=use_dcn)
        self.stage3_3 = IdentityBlock([128, 128, 512], use_dcn=use_dcn)

        # stage4
        self.stage4_0 = ConvBlock([256, 256, 1024], use_dcn=use_dcn)
        k = 21
        if depth == 50:
            k = 4
        self.stage4_layers = []
        for i in range(k):
            ly = IdentityBlock([256, 256, 1024], use_dcn=use_dcn)
            self.stage4_layers.append(ly)
        self.stage4_last_layer = IdentityBlock([256, 256, 1024], use_dcn=use_dcn)

        # stage5
        self.stage5_0 = ConvBlock([512, 512, 2048], use_dcn=use_dcn)
        self.stage5_1 = IdentityBlock([512, 512, 2048], use_dcn=use_dcn)
        self.stage5_2 = IdentityBlock([512, 512, 2048], use_dcn=use_dcn)


    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.maxpool(x)

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        for ly in self.stage4_layers:
            x = ly(x)
        s16 = self.stage4_last_layer(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)
        return [s4, s8, s16, s32]


