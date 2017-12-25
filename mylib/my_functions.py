import numpy as np
import chainer
from env import xp


def get_batch(ds, index, repeat):
    nt = ds.num_target
    # print(index)
    batch_size = index.shape[0]
    return_x = np.empty((batch_size, 3, 256, 256))
    return_t = np.zeros((batch_size, nt))
    for bi in range(batch_size):
        return_x[bi] = ds[index[bi]][0]
        return_t[bi][ds[index[bi]][1]] = 1
    return_x = return_x.reshape(batch_size, 3, 256, 256).astype(np.float32)
    return_t = return_t.astype(np.float32)
    return_x = chainer.Variable(xp.asarray(xp.tile(return_x, (repeat, 1, 1, 1))))
    return_t = chainer.Variable(xp.asarray(xp.tile(return_t, (repeat, 1))))
    return return_x, return_t
