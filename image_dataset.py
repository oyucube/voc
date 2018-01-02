import numpy as np
from PIL import Image, ImageOps
import csv
import chainer
import os


class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, file_name):
        self.path = path
        pairs = []
        n = 20
        with open('data/label_' + file_name + '.txt', 'r') as f:
            buf = f.read()
        txts = buf.splitlines()
        for txt in txts:
            buf = txt.split()
            if len(buf) == 21:
                file_path = "data/images/" + buf[0] + ".jpg"
                label = np.zeros(20)
                for i in range(n):
                    label[i] = int(buf[i + 1])
                pairs.append([file_path, label])
            else:
                # print(len(buf))
                break
        self._pairs = pairs
        self.len = len(self._pairs)
        self.num_target = n

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        filename = self._pairs[i][0]
        image = Image.open(filename)
        image = image.convert("RGB")
        if chainer.config.train:
            r = np.random.randint(32, size=3)
            if r[0] > 15:
                image = ImageOps.mirror(image)
            image = image.crop((r[1], r[2], r[1] + 224, r[2] + 224))
        image = image.resize((256, 256))
        image_array = np.asarray(image)
        image_array = image_array.astype('float32')
        image_array = image_array / 255
        image_array = image_array.transpose(2, 0, 1)  # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label

    def get_image(self, i):
        filename = self._pairs[i][0]
        image = Image.open(filename)
        image = image.convert("RGB")
        image = image.resize((256, 256))
        return image

    def d_print(self, i, crop=True):
        filename = self._pairs[i][0]
        image = Image.open(filename)
        image = image.convert("RGB")
        if crop:
            r = np.random.randint(32, size=3)
            if r[0] > 15:
                image = ImageOps.mirror(image)
            image = image.crop((r[1], r[2], r[1] + 224, r[2] + 224))
            image = image.resize((256, 256))
        image.show()


class CombDataset(ImageDataset):
    def __init__(self, path, file_name):
        self.path = path
        pairs = []
        n = 20
        class_counter = np.zeros(31)
        object_counter = np.zeros(20)
        with open('data/label_' + file_name + '.txt', 'r') as f:
            buf = f.read()
        txts = buf.splitlines()
        for txt in txts:
            buf = txt.split()
            if len(buf) == 21:
                file_path = "data/images/" + buf[0] + ".jpg"
                label = np.zeros(20)
                for i in range(n):
                    label[i] = int(buf[i + 1])
                # 7 cat 9 cow 11 dog 12 horse 16 sheep
                if (label[7] == 1) | (label[9] == 1) | (label[11] == 1) | (label[12] == 1) | (label[16] == 1):
                    c_label = 16 * label[7] + 8 * label[9] + 4 * label[11] + 2 * label[12] + label[16] - 1
                    a_label = np.zeros(31)
                    class_counter[int(c_label)] += 1
                    pairs.append([file_path, a_label])
            else:
                # print(len(buf))
                break
        print(class_counter)
        print(object_counter)
        self._pairs = pairs
        self.len = len(self._pairs)
        self.num_target = n