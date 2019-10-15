import numpy as np


def unique_for_raw(a):
    return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))) \
        .view(a.dtype).reshape(-1, a.shape[1])


def prune(a_vector, bs):
    index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
    return a_vector[index]
