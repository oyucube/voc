# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 04:46:24 2016

@author: oyu
"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import make_sampled_image
from env import xp
from modelfile.bnlstm import BNLSTM


class BASE(chainer.Chain):
    def __init__(self):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            cnn_1_1=L.Convolution2D(3, 64, 3, pad=1),  # in 32 out 16
            cnn_1_2=L.Convolution2D(64, 64, 3, pad=1),  # in 32 out 16
            cnn_2_1=L.Convolution2D(64, 128, 3, pad=1),  # in 16 out 12
            cnn_2_2=L.Convolution2D(128, 128, 3, pad=1),  # in 16 out 12
            cnn_3_1=L.Convolution2D(128, 256, 3, pad=1),  # in 12 out 8
            cnn_3_2=L.Convolution2D(256, 256, 3, pad=1),  # in 12 out 8
            cnn_3_3=L.Convolution2D(256, 256, 3, pad=1),  # in 12 out 8
            full_1=L.Linear(None, 256),
            full_2=L.Linear(None, 10),

            norm_1_1=L.BatchNormalization(64),
            norm_1_2=L.BatchNormalization(64),
            norm_2_1=L.BatchNormalization(128),
            norm_2_2=L.BatchNormalization(128),
            norm_3_1=L.BatchNormalization(256),
            norm_3_2=L.BatchNormalization(256),
            norm_3_3=L.BatchNormalization(256),
            norm_f1=L.BatchNormalization(256)
        )

    def __call__(self, x, target):
        h = F.relu(self.norm_1_1(self.cnn_1_1(x)))
        h = F.relu(self.norm_1_2(F.max_pooling_2d(self.cnn_1_2(h), 2, stride=2)))
        h = F.relu(self.norm_2_1(self.cnn_2_1(h)))
        h = F.relu(self.norm_2_2(F.max_pooling_2d(self.cnn_2_2(h), 2, stride=2)))
        h = F.relu(self.norm_3_1(self.cnn_3_1(h)))
        h = F.relu(self.norm_3_2(self.cnn_3_2(h)))
        h = F.relu(self.norm_3_3(F.max_pooling_2d(self.cnn_3_3(h), 2, stride=2)))
        h = F.relu(self.norm_f1(self.full_1(h)))

        y = self.full_2(h)
        if chainer.config.train:
            loss = F.softmax_cross_entropy(y, target)
            return loss
        else:
            y = F.softmax(y)
            index = xp.array(range(y.data.shape[0]))
            acc = y.data[index, target]
            acc = chainer.cuda.to_cpu(acc)
            return acc.sum()
