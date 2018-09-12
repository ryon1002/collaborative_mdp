import numpy as np
from . import util
import itertools


class POMDP(object):
    def __init__(self, s, a, z):
        self.s = s
        self.a = a
        self.z = z
        self.t = np.zeros((self.a, self.s, self.s))
        self.r = np.zeros((self.a, self.s))
        self.o = np.zeros((self.a, self.s, self.z))
        self._set_tro()
        self._pre_calc()

    def _pre_calc(self):
        pass

    def _set_tro(self):
        pass

    def calc_a_vector(self, d=1, bs=None, with_a=True):
        if d == 1:
            self.a_vector = self.r[:, :].copy()
            return
        self.calc_a_vector(d - 1, bs, False)
        a_vector = {}
        for a in range(self.a):
            p_a_vector = []
            p_a_vector_nums = []
            for z in range(self.z):
                p_a_vector.append(np.dot(self.a_vector, self.update[a, z].T))
                p_a_vector_nums.append(len(p_a_vector[-1]))

            a_vector_a = np.zeros((np.prod(p_a_vector_nums), self.s))
            for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
            a_vector_a = util.unique_for_raw(a_vector_a)
            a_vector[a] = self.r[a, :] + a_vector_a
        if with_a:
            self.a_vector_a = {a: util.prune(vector, bs) for a, vector in
                               a_vector.items()} if bs is not None else a_vector
        else:
            self.a_vector = util.prune(np.concatenate(list(a_vector.values()), axis=0),
                                       bs) if bs is not None else a_vector

    def value_a(self, a, b):
        return np.max(np.dot(self.a_vector_a[a], b))

    def value(self, b):
        return np.max(np.dot(self.a_vector, b))

    def get_best_action(self, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a.viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
