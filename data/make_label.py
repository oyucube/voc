import numpy as np
import os

target = "trainval"
# target = "trainval"
# target = "val"

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
          "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


with open("data/Main/" + target + ".txt", "r") as f:
    buf = f.read()
files = buf.splitlines()
label_files = []
for label_word in labels:
    with open("data/Main/" + label_word + "_" + target + ".txt", "r") as f:
        buf = f.read()
        label_files.append(buf.splitlines())
lines = 0
f = open("data/label_" + target + ".txt", "w")
c_0 = 0
c_1 = 0
c_minus = 0
for file in files:
    f.write(file)
    label_id = 0
    for label in label_files:
        txt = label[lines].split()
        if file == txt[0]:
            if txt[1] == "0":
                c_0 += 1
#                print("0 id detected.in {} line {} file{}".format(labels[label_id], lines, file))
                f.write(" 0")
            elif txt[1] == "1":
                c_1 += 1
                f.write(" 1")
            else:
                f.write(" 0")
            # f.write(" " + txt[1])
        else:
            print("error: file{} lines{}".format(file, lines))
            exit(0)
        label_id += 1
    lines += 1
    f.write("\n")

    # debug option
    # if lines == 10:
    #     break

print("one:{} zero:{}".format(c_1, c_0))
