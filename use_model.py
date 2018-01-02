# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# 乱数のシード固定
#
# i = np.random()
# np.random.seed()
import argparse
import chainer
from chainer import cuda, serializers
import importlib
import image_dataset
import socket
from mylib.my_functions import get_batch, draw_attention, label_id_to_str
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont




#  引数分解
parser = argparse.ArgumentParser()
# load model id

# * *********************************************    config    ***************************************************** * #
parser.add_argument("-a", "--am", type=str, default="model_at",
                    help="attention model")
parser.add_argument("-l", "--l", type=str, default="test1",
                    help="load model name")
test_b = 10
num_step = 2

# * **************************************************************************************************************** * #

# hyper parameters
parser.add_argument("-e", "--epoch", type=int, default=30,
                    help="iterate training given epoch times")
parser.add_argument("-b", "--batch_size", type=int, default=20,
                    help="batch size")
parser.add_argument("-m", "--num_l", type=int, default=30,
                    help="a number of sample ")
parser.add_argument("-s", "--step", type=int, default=2,
                    help="look step")
parser.add_argument("-v", "--var", type=float, default=0.02,
                    help="sample variation")
parser.add_argument("-g", "--gpu", type=int, default=-1,
                    help="use gpu")
# log config
parser.add_argument("-o", "--filename", type=str, default="",
                    help="prefix of output file names")
parser.add_argument("-p", "--logmode", type=int, default=1,
                    help="log mode")
args = parser.parse_args()

file_id = args.filename
num_lm = args.num_l
n_epoch = args.epoch
train_b = args.batch_size
train_var = args.var
gpu_id = args.gpu
crop = 1

# naruto ならGPUモード
if socket.gethostname() == "chainer":
    gpu_id = 0
    log_dir = "/home/y-murata/storage/voc/"
else:
    log_dir = "log/"
train_dataset = image_dataset.ImageDataset("voc/data/", "train")
val_dataset = image_dataset.ImageDataset("voc/data/", "val")

xp = cuda.cupy if gpu_id >= 0 else np

data_max = train_dataset.len
test_max = val_dataset.len
num_val = 1000
num_val_loop = 10  # val loop 10 times
#
# data_max = 1000
# test_max = 1000
# num_val = 100
# num_val_loop = 1  # val loop 10 times


img_size = 256
n_target = train_dataset.num_target
num_class = n_target
target_c = ""
# test_b = test_max

# モデルの作成
model_file_name = args.am

sss = importlib.import_module("modelfile." + model_file_name)
model = sss.SAF(n_out=n_target, img_size=img_size, var=train_var, n_step=num_step, gpu_id=gpu_id)



# model load
if len(args.l) != 0:
    print("load model model/best{}.model".format(args.l))
    serializers.load_npz('model/best_' + args.l + '.model', model)
else:
    print("load model!!!")
    exit()
# model load
# if len(args.l) != 0:
#     print("load model model/my{}{}.model".format(args.l, model_id))
#     serializers.load_npz('model/my' + args.l + model_id + '.model', model)

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# gpuの設定
if gpu_id >= 0:
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()


perm = np.random.permutation(test_max)
with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
    x, t = get_batch(val_dataset, perm[0:test_b], 1)
    y, ap, l_list, s_list = model.use_model(x, t)

print("ap")
print(ap)
print("l_list")
print(l_list)
print("s_list")
print(s_list)
for i in range(test_b):
    save_filename = "buf/attention" + str(i)
    # print(acc[i])
    img = val_dataset.get_image(perm[i])
    acc_str = ("AP:{:1.8f}".format(ap[i]))
    print(acc_str)
    bt = t.data[i]
    targets = np.where(bt == 1)[0]
    top5 = y[i].argsort()[::-1][:5]
    print(type(targets))
    print("top5:{} target:{}\n\n".format(top5, targets))
    # print(l_list)
    txt = []
    txt.append(acc_str)
    txt.append("top5 class      score")
    for j in range(5):
        txt.append(label_id_to_str(top5[j]) + "      {:1.3f}".format(y[i][top5[j]]))
    txt.append("target_class:")
    t_buf = ""
    for ttt in targets:
        t_buf += label_id_to_str(ttt) + " "
    txt.append(t_buf)
    draw_attention(img, l_list, s_list, i, save=save_filename, txt_buf=txt)
