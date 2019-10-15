import numpy as np


def _max_q_prob(arr):
    ret = (arr == np.max(arr)).astype(np.int)
    return ret / np.sum(ret)


def _exp_q_prob(arr, beta=1):
    ret = np.exp(arr * beta)
    if np.sum(ret) == 0:
        ret = np.ones_like(ret)
    return ret / np.sum(ret)

def exp_prob(arr, beta=1):
    ret = np.exp(arr * beta)
    if np.sum(ret) == 0:
        ret = np.ones_like(ret)
    return ret / np.sum(ret)

def _avg_prob(arr):
    if np.sum(arr) == 0:
        return arr
    return arr / np.sum(arr)
