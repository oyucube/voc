import numpy as np
import chainer
from env import xp
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_batch(ds, index, repeat):
    nt = ds.num_target
    # print(index)
    batch_size = index.shape[0]
    return_x = np.empty((batch_size, 3, 256, 256))
    return_t = np.zeros((batch_size, nt))
    for bi in range(batch_size):
        return_x[bi] = ds[index[bi]][0]
        return_t[bi] = ds[index[bi]][1]
    return_x = return_x.reshape(batch_size, 3, 256, 256).astype(np.float32)
    return_t = return_t.astype(np.float32)
    return_x = chainer.Variable(xp.asarray(xp.tile(return_x, (repeat, 1, 1, 1))))
    return_t = chainer.Variable(xp.asarray(xp.tile(return_t, (repeat, 1))))
    return return_x, return_t


def draw_attention(d_img, d_l_list, d_s_list, index, save="", txt_buf=[]):
    draw = ImageDraw.Draw(d_img)
    color_list = ["red", "yellow", "blue", "green"]
    size = 256
    for j in range(d_l_list.shape[0]):
        l = d_l_list[j][index]
        s = d_s_list[j][index]
        print(l)
        p1 = (size * (l - s / 2))
        p2 = (size * (l + s / 2))
        p1[0] = size - p1[0]
        p2[0] = size - p2[0]
        print([p1[0], p1[1], p2[0], p2[1]])
        draw.rectangle([p1[0], p1[1], p2[0], p2[1]], outline=color_list[j])
    # if len(acc) > 0:
    #     font = ImageFont.truetype("C:\\Windows\\Fonts\\msgothic.ttc", 20)
    #     draw.text([120, 230], acc, font=font, fill="red")
    if len(save) > 0:
        plt.plot()
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(d_img)
        plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        plt.tick_params(labelleft="off", left="off")  # y軸の削除

        plt.subplot(1, 2, 2)
        i = 0
        for b in txt_buf:
            i += 1
            plt.text(0.05, 1 - i * 0.1, b, fontsize=12)

        plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
        plt.tick_params(labelleft="off", left="off")  # y軸の削除
        plt.savefig(save + ".png")
        plt.close("all")
#         d_img.save(save + ".png")


def label_id_to_str(label_id):
    label_text = ["aeroplane  ", "bicycle    ", "bird       ", "boat       ", "bottle     ", "bus        ",
                  "car        ", "cat        ", "chair      ", "cow        ", "diningtable", "dog        ",
                  "horse      ", "motorbike  ", "person     ", "pottedplant", "sheep      ", "sofa       ",
                  "train      ", "tvmonitor  "]
    return label_text[label_id]
