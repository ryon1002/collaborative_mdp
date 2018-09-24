import numpy as np
import itertools
from . import util


class CoopIRL(object):
    def __init__(self, s, a_r, a_h, th):
        self.s = s
        self.a_r = a_r
        self.a_h = a_h
        self.th = th
        self.t = np.zeros((self.a_r, self.a_h, self.s, self.s))
        self.r = np.zeros((self.a_r, self.a_h, self.s, self.th))
        # self.o = np.zeros((self.th, self.a_r, self.s, self.a_h))
        self._set_tro()
        self._pre_calc()

    def _pre_calc(self):
        self.sum_r = np.zeros((self.a_r, self.s, self.th))
        for th in range(self.th):
            for s in range(self.s):
                self.sum_r[:, s, th] = np.max(self.r[:, :, s, th], axis=1)

        self.ns = {
            s: {a_r: {a_h: self._ex_all_nx(s, a_r, a_h) for a_h in range(self.a_h)} for a_r in
                range(self.a_r)} for s in range(self.s)}

    def _ex_all_nx(self, s, a_r, a_h):
        arr = self.t[a_r, a_h, s]
        ids = np.where(arr > 0)[0]
        return [i for i in zip(ids, arr[ids])]

    def _set_tro(self):
        pass

    def calc_a_vector(self, d=1, bs=None, with_a=True):
        if d == 1:
            self.a_vector = {s: util.prune(self.sum_r[:, s, :].copy(), bs)
                             for s in range(self.s)}
            return
        self.calc_a_vector(d - 1, bs, False)
        a_vector = {}

        for s in range(self.s):
            a_vector[s] = {}
            for a_r in range(self.a_r):
                a_vector_a = np.empty((0, self.th))
                for a_h in range(self.a_h):
                    for ns, _p in self.ns[s][a_r][a_h]:
                        a_vector_a = np.concatenate([a_vector_a, self.r[a_r, a_h, s] +
                                                     self.a_vector[ns]])
                a_vector[s][a_r] = util.unique_for_raw(np.max(a_vector_a, axis=0, keepdims=True))
        if with_a:
            self.a_vector_a = {s: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()}
                               for s, vectorA in a_vector.items()} if bs is not None else a_vector
        else:
            self.a_vector = {s: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                             s, vector in a_vector.items()} if bs is not None else a_vector

    def value_a(self, s, a_r, b):
        return np.max(np.dot(self.a_vector_a[s][a_r], b))

    def value(self, s, b):
        return np.max(np.dot(self.a_vector[s], b))

    def get_best_action(self, s, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
