import chainer
from modelfile.model_at import BASE
from chainer import Variable
from env import xp
import make_sampled_image
import chainer.functions as F
import chainer.links as L
from modelfile.bnlstm import BNLSTM
from chainer.links import VGG16Layers
from mylib.my_functions import np_to_img, img_to_np


class SAF(BASE):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(BASE, self).__init__(
            # the size of the inputs to each layer will be inferred
            # glimpse network
            # 切り取られた画像を処理する部分　位置情報 (glimpse loc)と画像特徴量の積を出力
            # in 256 * 256 * 3
            g_full=L.Linear(4096, 256),
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
            baseline=L.Linear(n_units, 1),

            class_full=L.Linear(n_units, n_out)
        )

        #
        # img parameter
        #
        self.vgg_model = VGG16Layers()
        if gpu_id == 0:
            self.use_gpu = True
            self.vgg_model.to_gpu()
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
        if self.use_gpu:
            x = chainer.cuda.to_cpu(x)
        img_list = []
        for i in range(x.shape[0]):
            img = np_to_img(x[i])
            img.resize((224, 224))
            img_list.append(img)
        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            f = self.vgg_model.extract(img_list, layers=["fc7"])["fc7"]
        return F.relu(self.g_full(Variable(f.data)))

