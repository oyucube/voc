import numpy as np
from PIL import Image, ImageOps
import csv
import chainer
import os
from dataset.base import ImageDataset


def np_to_img(array):
    img = array.transpose(1, 2, 0)
    pil = Image.fromarray(np.uint8(img * 256))
    return pil


def img_to_np(img):
    image_array = np.asarray(img)
    image_array = image_array.astype('float32')
    image_array = image_array / 255
    image_array = image_array.transpose(2, 0, 1)  # order of rgb / h / w
    return image_array


class MyDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, type, debug_limit=0):
        pairs = []
        self.path = path
        (train_data, val_data) = chainer.datasets.get_cifar10()
        if type == "train":
            data = train_data
            np.random.seed(70)
        elif type == "val":
            data = val_data
            np.random.seed(77)
        else: # error
            return
        if debug_limit != 0:
            num = debug_limit
        else:
            num = len(data)
        pos_rand = np.random.randint(0, 192, (num, 2))
        for i in range(num):
            black = np.zeros((3, 256, 256)).astype(np.float32)
            label = np.zeros(10)
            img, label_index = data[i]
            double = np_to_img(img).resize((64, 64))
            black[:, pos_rand[i][0]:pos_rand[i][0]+64, pos_rand[i][1]:pos_rand[i][1]+64] = img_to_np(double)
            label[label_index] = 1
            pairs.append([black, label.astype(np.float32)])
        self._pairs = pairs
        self.num_target = 10

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        return self._pairs[i][0], self._pairs[i][1]

    def get_image(self, i):
        return np_to_img(self._pairs[i][0])


data_name = "black + cifar10 "
