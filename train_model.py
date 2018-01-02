# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 04:46:33 2016

@author: oyu
"""
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0" #ここでオフに  オンにしたかったら1にするかコメントアウト
import numpy as np
# i = np.random()
# np.random.seed()
import argparse
import chainer
from chainer import cuda, serializers
from tqdm import tqdm
import datetime
import importlib
import image_dataset
import socket
import gc
from mylib.my_functions import get_batch
from mylib.my_logger import LOGGER

parser = argparse.ArgumentParser()

# model selection
parser.add_argument("-a", "--am", type=str, default="model_at",
                    help="attention model")
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
# load model id
parser.add_argument("-l", "--l", type=str, default="",
                    help="load model name")
# log config
parser.add_argument("-o", "--filename", type=str, default="",
                    help="prefix of output file names")
parser.add_argument("-p", "--logmode", type=int, default=1,
                    help="log mode")
args = parser.parse_args()

file_id = args.filename
num_lm = args.num_l
n_epoch = args.epoch
num_step = args.step
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

# log setting
if file_id == "":
    file_id = datetime.datetime.now().strftime("%m%d%H%M%S")
log_dir = log_dir + file_id + "/"
os.mkdir(log_dir)
logger = LOGGER(log_dir, file_id, n_epoch=n_epoch)

logger.l_print("{} class recognition\nclass:{} use VOC 2012 data set".format(num_class, target_c))
logger.l_print("model:{}".format(model_file_name))
logger.l_print("parameter\n")
logger.l_print("step:{} sample:{} batch_size:{} var:{} crop:{}".format(num_step, num_lm, train_b, train_var, crop))
logger.l_print("log dir:{}".format(log_dir))
logger.l_print("going to train {} epoch".format(n_epoch))
logger.update_log()

# train
for epoch in range(n_epoch):
    print("(epoch: {})\n".format(epoch + 1))
    perm = np.random.permutation(data_max)
    for i in tqdm(range(0, data_max, train_b), ncols=60):
        model.cleargrads()
        x, t = get_batch(train_dataset, perm[i:i+train_b], num_lm)
        loss_func = model(x, t, mode=1)
        logger.set_loss(loss_func.data, epoch)
        loss_func.backward()
        loss_func.unchain_backward()  # truncate
        optimizer.update()
        del x, t, loss_func
        gc.collect()

    # evaluate
    train_map = 0
    val_map = 0
    perm_train = np.random.permutation(data_max)
    perm_val = np.random.permutation(test_max)
    for i in range(0, num_val, int(num_val / num_val_loop)):
        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            x, t = get_batch(val_dataset, perm_val[i:i + 100], 1)
            train_map += model.mean_average_perception(x, t)
            x, t = get_batch(train_dataset, perm_train[i:i + 100], 1)
            val_map += model.mean_average_perception(x, t)

    # save accuracy
    logger.set_acc(train_map / num_val_loop, val_map / num_val_loop, epoch)
    logger.save_acc()
    logger.update_log()
    # save model
    if gpu_id >= 0:
        serializers.save_npz(log_dir + "/" + logger.best + file_id + '.model', model.to_cpu())
        model.to_gpu()
    else:
        serializers.save_npz(log_dir + "/" + logger.best + file_id + '.model', model)

# logger.l_print("last acc:{}  max_acc:{}\n".format(acc1_array[n_epoch - 1], max_acc))
