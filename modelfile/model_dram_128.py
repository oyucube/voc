import chainer
from chainer import Variable
from env import xp
import make_sampled_image
import chainer.functions as F
import chainer.links as L


class SAF(chainer.Chain):
    def __init__(self, n_units=256, n_out=0, img_size=112, var=0.18, n_step=2, gpu_id=-1):
        super(SAF, self).__init__(
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
            # full_2=L.Linear(None, 10),

            glimpse_loc=L.Linear(2, 256),

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

    def reset(self):
        self.rnn_1.reset_state()
        self.rnn_2.reset_state()

    def __call__(self, x, target):
        self.reset()
        n_step = self.n_step
        num_lm = x.shape[0]
        if chainer.config.train:
            r_buf = 0
            l, b = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm = self.make_img(x, l, num_lm, random=1)
                    l1, y, b1 = self.recurrent_forward(xm, lm)

                    loss, size_p = self.cul_loss(y, target, l, lm)
                    r_buf += size_p
                    r = xp.where(
                        xp.argmax(y.data, axis=1) == xp.argmax(target, axis=1), 1, 0).reshape((num_lm, 1)).astype(
                        xp.float32)
                    loss *= self.r_recognize
                    loss += F.sum((r - b) * (r - b))  # loss baseline
                    k = self.r * (r - b.data)  # calculate r
                    loss += F.sum(k * r_buf)

                    return loss / num_lm
                else:
                    xm, lm = self.make_img(x, l, num_lm, random=1)
                    l1, y, b1 = self.recurrent_forward(xm, lm)
                    loss, size_p = self.cul_loss(y, target, l, lm)
                    r_buf += size_p
                l = l1
                b = b1

        else:
            l, b1 = self.first_forward(x, num_lm)
            for i in range(n_step):
                if i + 1 == n_step:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)
                    accuracy = xp.sum(y.data * target)
                    if self.use_gpu:
                        accuracy = chainer.cuda.to_cpu(accuracy)
                    return accuracy
                else:
                    xm, lm = self.make_img(x, l, num_lm, random=0)
                    l1, y, b = self.recurrent_forward(xm, lm)
                l = l1

    def use_model(self, x, t):
        self.reset()
        num_lm = x.shape[0]
        n_step = self.n_step
        s_list = xp.ones((n_step, num_lm, 1)) * (128 / self.img_size)
        l_list = xp.empty((n_step, num_lm, 2))
        l, b1 = self.first_forward(x, num_lm)
        for i in range(n_step):
            if i + 1 == n_step:
                xm, lm = self.make_img(x, l, num_lm, random=0)
                l1, y, b = self.recurrent_forward(xm, lm)
                l_list[i] = l1.data
                accuracy = y.data * t
                return xp.sum(accuracy, axis=1), l_list, s_list
            else:
                xm, lm = self.make_img(x, l, num_lm, random=0)
                l1, y, b = self.recurrent_forward(xm, lm)
            l = l1
            l_list[i] = l.data
        return

    def first_forward(self, x, num_lm):
        self.rnn_1(Variable(xp.zeros((num_lm, self.n_unit)).astype(xp.float32)))
        h2 = F.relu(self.l_norm_cc1(self.context_cnn_1(F.average_pooling_2d(x, 4, stride=4))))
        h3 = F.relu(self.l_norm_cc2(self.context_cnn_2(h2)))
        h4 = F.relu(self.l_norm_cc3(self.context_cnn_3(F.max_pooling_2d(h3, 2, stride=2))))
        h5 = F.relu(self.l_norm_cc4(self.context_cnn_4(h4)))
        h6 = F.relu(self.l_norm_cc5(self.context_cnn_5(h5)))
        h7 = F.relu(self.context_full(F.max_pooling_2d(h6, 2, stride=2)))
        h8 = F.relu(self.rnn_2(h7))

        l = F.sigmoid(self.attention_loc(h8))
        b = F.sigmoid(self.baseline(Variable(h8.data)))
        return l, b

    def recurrent_forward(self, xm, lm):
        hgl = F.relu(self.glimpse_loc(lm))

        h = self.glimpse_forward(xm)
        hr1 = F.relu(self.rnn_1(hgl * h))

        hr2 = F.relu(self.rnn_2(hr1))
        l = F.sigmoid(self.attention_loc(hr2))
        y = F.softmax(self.class_full(hr1))
        b = F.sigmoid(self.baseline(Variable(hr2.data)))
        return l, y, b

    def glimpse_forward(self, x):
        h = F.relu(self.norm_1_1(self.cnn_1_1(x)))
        h = F.relu(self.norm_1_2(F.max_pooling_2d(self.cnn_1_2(h), 2, stride=2)))
        h = F.relu(self.norm_2_1(self.cnn_2_1(h)))
        h = F.relu(self.norm_2_2(F.max_pooling_2d(self.cnn_2_2(h), 2, stride=2)))
        h = F.relu(self.norm_3_1(self.cnn_3_1(h)))
        h = F.relu(self.norm_3_2(self.cnn_3_2(h)))
        h = F.relu(self.norm_3_3(F.max_pooling_2d(self.cnn_3_3(h), 2, stride=2)))
        h = F.relu(self.norm_f1(self.full_1(h)))
        return h

    def cifar10(self, x, target):
        h = self.glimpse_forward(x)
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

    # loss 関数を計算

    def cul_loss(self, y, target, l, lm):

        l1, l2 = F.split_axis(l, indices_or_sections=2, axis=1)
        m1, m2 = F.split_axis(lm, indices_or_sections=2, axis=1)
        loss_loc = ((l1 - m1) * (l1 - m1) + (l2 - m2) * (l2 - m2)) / self.var / 2
        # size

        accuracy = y * target

        loss = -F.sum(accuracy)
        return loss, loss_loc

    def make_img(self, x, l, num_lm, random=0):
        s = xp.log10(xp.ones((1, 1)) * 128 / 256) + 1
        sm = xp.repeat(s, num_lm, axis=0)

        if random == 0:
            lm = Variable(xp.clip(l.data, 0, 1))
        else:
            eps = xp.random.normal(0, 1, size=l.data.shape).astype(xp.float32)
            lm = xp.clip(l.data + eps * xp.sqrt(self.vars), 0, 1)
            lm = Variable(lm.astype(xp.float32))
        if self.use_gpu:
            xm = make_sampled_image.generate_xm_rgb_gpu(lm.data, sm, x, num_lm, g_size=self.gsize)
        else:
            xm = make_sampled_image.generate_xm_rgb(lm.data, sm, x, num_lm, g_size=self.gsize)
        return xm, lm
