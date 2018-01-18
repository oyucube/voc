import chainer
import numpy as np
(train, val) = chainer.datasets.get_cifar10()

test = chainer.dataset.concat_examples(train[0:3])
perm = np.random.permutation(4)
train_b = 2
for i in range(0, 4, train_b):
    x, t = chainer.dataset.concat_examples(train[2,4,6])
    print(x)
    print(t)
