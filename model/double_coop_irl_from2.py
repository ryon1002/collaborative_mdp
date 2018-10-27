import numpy as np
import itertools
from . import util


class CoopIRL(object):
    def __init__(self, s, a_r, a_h, th_r, th_h):
        self.s = s
        self.a_r = a_r
        self.a_h = a_h
        self.th_r = th_r
        self.th_h = th_h
        self.t = np.zeros((self.a_r, self.a_h, self.s, self.s))
        self.r = np.zeros((self.a_r, self.a_h, self.s, self.th_r, self.th_h))
        # self.o = np.zeros((self.th, self.a_r, self.s, self.a_h))
        self._set_tro()
        self._pre_calc()

    def func(self, arr):
        ret = np.zeros_like(arr)
        ret[np.argmax(arr)] = 1
        return ret

    def _max_q_prob(self, arr):
        ret = (arr == np.max(arr)).astype(np.int)
        return ret / np.sum(ret)

    def _avg_prob(self, arr):
        if np.sum(arr) == 0:
            return arr
        return arr / np.sum(arr)

    def _pre_calc(self):
        self.sum_r = np.zeros((self.a_r, self.s, self.th_r, self.th_h))
        for th_r in range(self.th_r):
            for th_h in range(self.th_h):
                for s in range(self.s):
                    self.sum_r[:, s, th_r, th_h] = np.max(self.r[:, :, s, th_r, th_h], axis=1)

        self.ns = {
            s: {a_r: {a_h: self._ex_all_nx(s, a_r, a_h) for a_h in range(self.a_h)} for a_r in
                range(self.a_r)} for s in range(self.s)}


    def _ex_all_nx(self, s, a_r, a_h):
        arr = self.t[a_r, a_h, s]
        ids = np.where(arr > 0)[0]
        return [i for i in zip(ids, arr[ids])]

    def _set_tro(self):
        pass

    def calc_a_vector(self, d, bs=None, with_a=True):
        if d == 1:
            self.a_vector = {}
            for s in range(self.s):
                self.a_vector[s] = {}
                for th_r in range(self.th_r):
                    self.a_vector[s][th_r] = np.zeros((1, self.th_h))
                    for th_h in range(self.th_h):
                        self.a_vector[s][th_r][0][th_h] = np.max(self.r[:, :, s, th_r, th_h])
            return
        self.calc_a_vector(d - 1, bs, False)

        a_vector = {s: {th_r: {} for th_r in range(self.th_r)} for s in range(self.s)}

        for s in range(self.s):
            r_val = np.zeros((self.a_r, self.th_r))
            for th_r in range(self.th_r):
                for a_r in range(self.a_r):
                    tmp_th_h_val = np.zeros(self.th_h)
                    for th_h in range(self.th_h):
                        tmp_a_h_val = np.zeros(self.a_h)
                        for a_h in range(self.a_h):
                            for ns, _p in self.ns[s][a_r][a_h]:
                                tmp_a_h_val[a_h] += np.max(self.a_vector[ns][th_r][:, th_h])
                        tmp_th_h_val[th_h] = np.max(tmp_a_h_val)
                    r_val[a_r, th_r] = np.mean(tmp_th_h_val)
            r_pi = np.apply_along_axis(self._max_q_prob, 0, r_val)
            inv_r_pi = np.apply_along_axis(self._max_q_prob, 1, r_pi)

        for th_r in range(self.th_r):
            for s in range(self.s):
                for a_r in range(self.a_r):
                    q_vector_2 = np.zeros((self.a_h, self.th_h))
                    for a_h in range(self.a_h):
                        for th_r2 in range(self.th_r):
                            q_vector2_a = np.zeros((0, self.th_h))
                            for ns, _p in self.ns[s][a_r][a_h]:
                                q_vector2_a = np.concatenate([q_vector2_a, self.r[a_r, a_h, s, th_r2] +
                                                             self.a_vector[ns][th_r2]])
                            q_vector_2[a_h] += np.max(q_vector2_a * inv_r_pi[a_r, th_r2], axis=0)
                    pi = np.apply_along_axis(self._max_q_prob, 0, q_vector_2)

                    update = np.empty((self.a_h, self.s, self.th_h))
                    for th in range(self.th_h):
                        t = np.sum(self.t[a_r, :, s] * pi[:, th][:, np.newaxis], axis=0)
                        update[:, :, th] = np.outer(pi[:, th], t)

                    p_a_vector = []
                    p_a_vector_nums = []
                    for a_h in range(self.a_h):
                        tmp_p_a_vector = np.empty((0, self.th_h))
                        for ns, _p in self.ns[s][a_r][a_h]:
                            tmp_p_a_vector = np.concatenate(
                                [tmp_p_a_vector,
                                 self.a_vector[ns][th_r] * update[a_h, ns] +
                                 self.r[a_r, a_h, s, th_r, :] * pi[a_h, :]])
                        p_a_vector.append(util.unique_for_raw(tmp_p_a_vector))
                        p_a_vector_nums.append(len(p_a_vector[-1]))
                    a_vector_a = np.zeros((np.prod(p_a_vector_nums), self.th_h))
                    for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                        a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                    a_vector_a = util.unique_for_raw(a_vector_a)
                    # a_vector[s][a_r] = a_vector_a
                    a_vector[s][th_r][a_r] = a_vector_a

        if with_a:
            self.a_vector_a = {
                s: {th_r: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()} for
                    th_r, vectorA in th_vector.items()} for s, th_vector in
                a_vector.items()} if bs is not None else a_vector
            # print(self.a_vector_a[0])
        else:
            self.a_vector = {
                s: {th_r: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                    th_r, vector in th_vector.items()}
                for s, th_vector in a_vector.items()} if bs is not None else a_vector


    def value_a(self, s, th_r, a_r, b):
        return np.max(np.dot(self.a_vector_a[s][th_r][a_r], b))

    def value(self, s, th_r, b):
        return np.max(np.dot(self.a_vector[s][th_r], b))

    def get_best_action(self, s, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
