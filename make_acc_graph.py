import matplotlib.pyplot as plt
import numpy as np

acc_list = ["normal_train", "normal_test", "pretrain_train", "pretrain_test"]


plt.figure()
# plt.rcParams["font.size"] = 18
plt.xlim([0, 30])
# plt.ylim([0, 1])
for item in acc_list:
    acc = np.load("graph/" + item + ".npy")
    plt.plot(acc, label=item)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc='uppper right', bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0., )
#plt.legend(loc="lower left")
# plt.legend()
plt.savefig("graph/graph.png")
