import chainer
import numpy as np
from modelfile.cifar10 import BASE
from modelfile.model_at import SAF
from mylib.my_functions import copy_model
from chainer import cuda, serializers
from dataset.common import MyDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# alpha = 64
# ds = MyDataset("voc/data/", "train")
#
# for i in range(10):
#     img = ds.get_image(i)
#
#     img = img.resize((256, 256))
#     img = img.resize((alpha, alpha))
#     img.show()
#
#

# 画像の表示
# model1 = SAF(n_out=2)
# model2 = SAF(n_out=2)
#
#
# copy_model(model1, model2)
# ds = MyDataset("voc/data/", "train", debug_limit=10)
# ds = MyDataset("voc/data/", "val", debug_limit=10)
#
# print(ds[0][0])
# img = ds[0][0].transpose(1, 2, 0)
# plt.imshow(img)
# plt.show()
# plt.savefig("test.png")
# from dataset.common import MyDataset
#
# ds = MyDataset("voc/data/", "train")
# ds = MyDataset("voc/data/", "val")

# model1 = BASE()
# model2 = SAF(n_out=2)
# serializers.load_npz('model/db.model', model2)
# serializers.load_npz('model/cifar84.model', model1)
#
# print(isinstance(model2, chainer.Chain))
#
# copy_model(model1, model2)
