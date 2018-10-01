import numpy as np
import itertools
from . import util


class CoopPOMDP(object):
    def __init__(self, s, a_r, a_h, th):
        self.s = s
        self.a_r = a_r
        self.a_h = a_h
        self.th = th
        self.t = np.zeros((self.a_r, self.a_h, self.s, self.s))
        self.r = np.zeros((self.a_r, self.a_h, self.s, self.th))
        self.o = np.zeros((self.th, self.a_r, self.s, self.a_h))
        self._set_tro()
        self._pre_calc()

    def _pre_calc(self):
        self.sum_r = np.zeros((self.a_r, self.s, self.th))
        for th in range(self.th):
            for s in range(self.s):
                self.sum_r[:, s, th] = np.sum(self.o[th, :, s] * self.r[:, :, s, th], axis=1)
        # P(x', a_h|x, a_r; theta)
        self.update = np.zeros((self.a_r, self.a_h, self.s, self.s, self.th))
        for th in range(self.th):
            for s in range(self.s):
                for a_r in range(self.a_r):
                    t = np.sum(self.t[a_r, :, s] * self.o[th, a_r, s][:, np.newaxis], axis=0)
                    self.update[a_r, :, s, :, th] = np.outer(self.o[th, a_r, s], t)
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
            self.a_vector = {s: self.sum_r[:, s, :].copy() for s in range(self.s)}
            return
        self.calc_a_vector(d - 1, bs, False)
        a_vector = {}
        for s in range(self.s):
            a_vector[s] = {}
            for a_r in range(self.a_r):
                p_a_vector = []
                p_a_vector_nums = []
                for a_h in range(self.a_h):
                    tmp_p_a_vector = np.empty((0, self.th))
                    for ns, _p in self.ns[s][a_r][a_h]:
                        tmp_p_a_vector = np.concatenate(
                            [tmp_p_a_vector, self.a_vector[ns] * self.update[ a_r, a_h, s, ns]])
                    p_a_vector.append(util.unique_for_raw(tmp_p_a_vector))
                    p_a_vector_nums.append(len(p_a_vector[-1]))
                a_vector_a = np.zeros((np.prod(p_a_vector_nums), self.th))
                for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                    a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                a_vector_a = util.unique_for_raw(a_vector_a)
                a_vector[s][a_r] = self.sum_r[a_r, s, :] + a_vector_a
        if with_a:
            self.a_vector_a = {s: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()}
                               for s, vectorA in
                               a_vector.items()} if bs is not None else a_vector
        else:
            self.a_vector = {s: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                             s, vector in
                             a_vector.items()} if bs is not None else a_vector
        for s in range(self.s):
            print(d, s, self.a_vector[s])

    def value_a(self, s, a_r, b):
        return np.max(np.dot(self.a_vector_a[s][a_r], b))

    def value(self, s, b):
        return np.max(np.dot(self.a_vector[s], b))

    def get_best_action(self, s, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
