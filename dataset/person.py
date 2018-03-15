import numpy as np
from PIL import Image, ImageOps
import csv
import chainer
import os
from dataset.base import ImageDataset


class MyDataset(ImageDataset):
    def __init__(self, path, file_name):
        self.path = path
        pairs = []
        target_counter = np.zeros(2)
        with open('data/label_' + file_name + '.txt', 'r') as f:
            buf = f.read()
        txts = buf.splitlines()
        if file_name == "train":
            stopper = 1994
        else:
            stopper = 2093
        for txt in txts:
            buf = txt.split()
            if len(buf) == 21:
                file_path = "data/images/" + buf[0] + ".jpg"
                label = np.zeros(2)  # 2class
                if int(buf[14 + 1]) == 1:
                    label[1] = 1
                    target_counter[0] += 1
                    pairs.append([file_path, label])
                else:
                    label[0] = 1
                    if target_counter[1] != stopper:
                        pairs.append([file_path, label])
                        target_counter[1] += 1
            else:
                # print(len(buf))
                break
        print(target_counter)
        print(len(pairs))
        self._pairs = pairs
        self.len = len(self._pairs)
        self.num_target = 2

    def get_example(self, i):
        filename = self._pairs[i][0]
        image = Image.open(filename)
        image = image.convert("RGB")
        image = image.resize((256, 256))
        if chainer.config.train:
            back = Image.fromarray(np.uint8(np.zeros((256, 256, 3))))
            r = np.random.randint(64, size=4)
            if r[0] > 32:
                image = ImageOps.mirror(image)
            image = image.resize((256 - r[1], 256 - r[1]))
            x = int(r[2] * r[1] / 64)
            y = int(r[3] * r[1] / 64)
            back.paste(image, (x, y))
            image = back
        image_array = np.asarray(image)
        image_array = image_array.astype('float32')
        image_array = image_array / 255
        image_array = image_array.transpose(2, 0, 1)  # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label


data_name = "person data "
