from modelfile.model_at import BASE
from chainer import Variable
from env import xp
import make_sampled_image
import chainer.functions as F
import chainer.links as L
from modelfile.bnlstm import BNLSTM


class SAF(BASE):
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
