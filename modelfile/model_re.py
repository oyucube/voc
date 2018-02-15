from modelfile.model_at import BASE
from chainer import Variable
from env import xp
import make_sampled_image
import chainer.functions as F
import chainer.links as L
from modelfile.bnlstm import BNLSTM


class SAF(BASE):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            # in 256 * 256 * 3
            cnn_1_1=L.Convolution2D(3, 64, 3, pad=1),
            cnn_1_2=L.Convolution2D(64, 64, 3, pad=1),
            cnn_2_1=L.Convolution2D(64, 128, 3, pad=1),
            cnn_2_2=L.Convolution2D(128, 128, 3, pad=1),
            cnn_3_1=L.Convolution2D(128, 256, 3, pad=1),
            cnn_3_2=L.Convolution2D(256, 256, 3, pad=1),
            cnn_3_3=L.Convolution2D(256, 256, 3, pad=1),
            full_1=L.Linear(4 * 4 * 256, 256),
            full_2=L.Linear(None, 10),

            glimpse_loc=L.Linear(3, 256),

            norm_1_1=L.BatchNormalization(64),
            norm_1_2=L.BatchNormalization(64),
            norm_2_1=L.BatchNormalization(128),
            norm_2_2=L.BatchNormalization(128),
            norm_3_1=L.BatchNormalization(256),
            norm_3_2=L.BatchNormalization(256),
            norm_3_3=L.BatchNormalization(256),
            norm_f1=L.BatchNormalization(256),

            # 記憶を用いるLSTM部分
            rnn_1=L.LSTM(n_units, n_units),
            rnn_2=L.LSTM(n_units, n_units),

            # 注意領域を選択するネットワーク
            attention_loc=L.Linear(n_units, 2),
            attention_scale=L.Linear(n_units, 1),

            # 入力画像を処理するネットワーク
            context_cnn_1=L.Convolution2D(3, 64, 3, pad=1),
            context_cnn_2=L.Convolution2D(64, 64, 3, pad=1),
            context_cnn_3=L.Convolution2D(64, 64, 3, pad=1),
            context_cnn_4=L.Convolution2D(64, 64, 3, pad=1),
            context_cnn_5=L.Convolution2D(64, 64, 3, pad=1),
            context_full=L.Linear(16 * 16 * 64, n_units),

            l_norm_cc1=L.BatchNormalization(64),
            l_norm_cc2=L.BatchNormalization(64),
            l_norm_cc3=L.BatchNormalization(64),
            l_norm_cc4=L.BatchNormalization(64),
            l_norm_cc5=L.BatchNormalization(64),

            # baseline network 強化学習の期待値を学習し、バイアスbとする
            baseline=L.Linear(n_units, 1)
        )

        #
        # img parameter
        #
        if gpu_id == 0:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.img_size = img_size
        self.gsize = 32
        self.train = True
        self.var = var
        self.vars = var
        self.n_unit = n_units
        self.num_class = n_out
        # r determine the rate of position
        self.r = 0.5
        self.r_recognize = 1.0
        self.n_step = n_step

    def glimpse_forward(self, x):
        self.zerograds()
        h = F.relu(self.norm_1_1(self.cnn_1_1(x)))
        h = F.relu(self.norm_1_2(F.max_pooling_2d(self.cnn_1_2(h), 2, stride=2)))
        h = F.relu(self.norm_2_1(self.cnn_2_1(h)))
        h = F.relu(self.norm_2_2(F.max_pooling_2d(self.cnn_2_2(h), 2, stride=2)))
        h = F.relu(self.norm_3_1(self.cnn_3_1(h)))
        h = F.relu(self.norm_3_2(self.cnn_3_2(h)))
        h = F.relu(self.norm_3_3(F.max_pooling_2d(self.cnn_3_3(h), 2, stride=2)))
        h = Variable(h.data)
        h = F.relu(self.norm_f1(self.full_1(h)))
        return h

    def recurrent_forward(self, xm, lm, sm):
        ls = xp.concatenate([lm.data, sm.data], axis=1)
        hgl = F.relu(self.glimpse_loc(Variable(ls)))

        h = self.glimpse_forward(xm)
        hr1 = F.relu(self.rnn_1(hgl * h))
        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        s = F.sigmoid(self.attention_scale(hr2))
        y = F.softmax(self.full_2(hgl))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, s, y, b