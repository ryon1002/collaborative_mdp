import numpy as np
import itertools


class POMDP(object):
    def __init__(self, s, a, z):
        self.s = s
        self.a = a
        self.z = z
        self.t = np.zeros((self.a, self.s, self.s))
        self.r = np.zeros((self.a, self.s))
        self.o = np.zeros((self.a, self.s, self.z))

    def pre_calc(self):
        # p(y | x, a, z) \propto p(z | a, x, y) * p(y | a, x)
        # and multiply p(z | a, x, y) in advance
        self.update = np.zeros((self.a, self.z, self.s, self.s))

        # for a in range(self.a):
        #     for z in range(self.z):
        #         self.update[a, z] = self.t[a] * self.o[a, :, z].T
        for s in range(self.s):
            for a in range(self.a):
                self.update[a, :, s] = np.outer(self.o[a, s], self.t[a, s])
                print(self.update[a, :, s])
        for s in range(self.s):
            for a in range(self.a):
                # p_z_as = np.dot(self.t[a, s], self.o[a, :])
                self.update[a, :, s] = np.outer(self.o[a, s], self.t[a, s])
                print(self.update[a, :, s])
        # exit()


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

            a_vector_xa = np.zeros((np.prod(p_a_vector_nums), self.s))
            for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                a_vector_xa[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
            a_vector_xa = self.unique_for_raw(a_vector_xa)
            # print(self.r[a, :])
            a_vector[a] = self.r[a, :] + a_vector_xa
        if with_a:
            self.a_vector_a = {a: self.prune(vector, bs) for a, vector in a_vector.items()} if bs is not None else a_vector
        else:
            self.a_vector = self.prune(np.concatenate(list(a_vector.values()), axis=0), bs) if bs is not None else a_vector

    @staticmethod
    def unique_for_raw(a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))) \
            .view(a.dtype).reshape(-1, a.shape[1])

    @staticmethod
    def prune(a_vector, bs):
        index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
        return a_vector[index]

    def value_a(self, a, b):
        return np.max(np.dot(self.a_vector_a[a], b))

    def value(self, b):
        # print(b[:, np.newaxis])
        # print(self.a_vector)
        return np.max(np.dot(self.a_vector, b))

    def get_best_action(self, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a.viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]

