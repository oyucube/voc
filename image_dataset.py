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
        with open('data/label' + file_name + '.txt', 'r') as f:
            buf = f.read()
        txts = buf.splitlines()
        for txt in txts:
            buf = txt.split()
            if len(buf) == 11:
                file_path = buf[0] + ".jpg"
                label = np.array(20)
                for i in range(n):
                    label[i + 1] = int(buf[i])
                pairs.append([file_path, label])
            else:
                print(len(buf))
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
        else:
            image_array = np.asarray(image)
        image_array = image_array.astype('float32')
        image_array = image_array / 255
        image_array = image_array.transpose(2, 0, 1)  # order of rgb / h / w
        label = np.int32(self._pairs[i][1])
        return image_array, label
