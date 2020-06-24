[English](README_en.md) | 简体中文

# Keras-SOLO

## 概述
Keras-SOLO, "It solo solo e-e-everybody, I do it solo.", 来自母胎solo的独奏(>_<)。

6G的卡也可训练，前提是必须要冻结网络前部分的层。

## 传送门

Keras版YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch版YOLOv3：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle版YOLOv3：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle完美复刻版版yolact: https://github.com/miemie2013/PaddlePaddle_yolact

yolov3魔改成yolact: https://github.com/miemie2013/yolact

Keras版YOLOv4: https://github.com/miemie2013/Keras-YOLOv4

Pytorch版YOLOv4: 制作中

Paddle镜像版YOLOv4：https://github.com/miemie2013/Paddle-YOLOv4

Keras版SOLO: https://github.com/miemie2013/Keras-SOLO

## 更新日记

2020/06/11:初次见面

2020/06/24:实现了DecoupledSOLO_R50_FPN

## 环境搭建

需要安装cuda9, tf1.12.2等，见requirements.txt。预计可能会升级到tf2、cuda10。

## 训练
去原作者的仓库https://github.com/WXinlong/SOLO 下载Decoupled_SOLO_R50_3x.pth，将该模型放在本项目根目录下，运行1_pytorch2keras.py得到一个预训练模型solo.h5，它也位于根目录下。
训练SOLO需要的显存比较大，所以train.py里我冻结了一部分层(ly.trainable = False)。
运行train.py进行训练。通过修改config.py代码来进行更换数据集、更改超参数以及训练参数。

追求更高的精度，你需要把冻结层的代码删除，也就是train.py中ly.trainable = False那一部分。但是需要你有一块高显存的显卡。
训练时默认每5000步计算一次验证集的mAP。

训练时如果发现mAP很稳定了，就停掉，修改学习率为原来的十分之一，接着继续训练，mAP还会再上升。暂时是这样手动操作。

## 评估
pass

## test-dev
pass

## 预测
python infer.py

## Citations
```
@article{wang2019solo,
  title={SOLO: Segmenting Objects by Locations},
  author={Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  journal={arXiv preprint arXiv:1912.04488},
  year={2019}
}
```

```
@article{wang2020solov2,
  title={SOLOv2: Dynamic, Faster and Stronger},
  author={Wang, Xinlong and Zhang, Rufeng and  Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={arXiv preprint arXiv:2003.10152},
  year={2020}
}
```

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。


## 广告位招租
有偿接私活，可联系微信wer186259，金主快点来吧！
