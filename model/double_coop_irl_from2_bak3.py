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

    def _exp_q_prob(self, arr):
        ret = np.exp(arr * 1)
        if np.sum(ret) == 0:
            ret = np.ones_like(ret)
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

    def calc_a_vector(self, d, bs=None, algo=1):
        if d == 1:
            self.a_vector = {}
            for s in range(self.s):
                self.a_vector[s] = {}
                for th_r in range(self.th_r):
                    self.a_vector[s][th_r] = np.zeros((1, self.th_h))
            self.beliefs = {}
            for th_r in range(self.th_r):
                self.beliefs[th_r] = {}
                for s in range(self.s):
                    self.beliefs[th_r][s] = np.array([0.5, 0.5])
            return
        self.calc_a_vector(d - 1, bs, algo)

        a_vector = {s: {th_r: {} for th_r in range(self.th_r)} for s in range(self.s)}

        check_s = 0
        inv_r_pi = {}
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
                            tmp_a_h_val[a_h] += self.r[a_r, a_h, s, th_r, th_h]
                        tmp_th_h_val[th_h] = np.max(tmp_a_h_val)

                    # r_val[a_r, th_r] = np.mean(tmp_th_h_val)
                    belief = self.beliefs[th_r][s] if s in self.beliefs[th_r] else np.array([0.5, 0.5])
                    r_val[a_r, th_r] = np.dot(tmp_th_h_val, belief)
            if algo == 1:
                # r_pi = np.apply_along_axis(self._exp_q_prob, 0, r_val / 10)  # full bayes
                r_pi = np.apply_along_axis(self._max_q_prob, 0, r_val)  # full bayes
                inv_r_pi[s] = np.apply_along_axis(self._max_q_prob, 1, r_pi)
            elif algo == 0:
                r_pi = np.apply_along_axis(self._max_q_prob, 1, r_val) # bias
                inv_r_pi[s] = np.apply_along_axis(self._max_q_prob, 1, r_pi)

        self.h_pi = {}
        for th_r in range(self.th_r):
            self.h_pi[th_r] = {}
            for s in range(self.s):
                if algo == 2:
                    inv_r_pi[s] = np.zeros((self.a_r, self.th_r))
                    inv_r_pi[s][:, th_r] = 1
                self.h_pi[th_r][s] = {}
                for a_r in range(self.a_r):
                    q_vector_2 = np.zeros((self.a_h, self.th_h))
                    for a_h in range(self.a_h):
                        for th_r2 in range(self.th_r):
                            q_vector2_a = np.zeros((0, self.th_h))
                            for ns, _p in self.ns[s][a_r][a_h]:
                                q_vector2_a = np.concatenate([q_vector2_a,
                                                              self.r[a_r, a_h, s, th_r2] +
                                                              self.a_vector[ns][th_r2]])
                            q_vector_2[a_h] += np.max(q_vector2_a * inv_r_pi[s][a_r, th_r2], axis=0)

                    pi = np.apply_along_axis(self._max_q_prob, 0, q_vector_2)
                    self.h_pi[th_r][s][a_r] = np.apply_along_axis(self._exp_q_prob, 1, pi)
                    # self.h_pi[th_r][s][a_r] = np.apply_along_axis(self._exp_q_prob, 1,
                    #                                               q_vector_2 / 1000)

                    update = np.empty((self.a_h, self.s, self.th_h))
                    for th in range(self.th_h):
                        t = np.sum(self.t[a_r, :, s] * pi[:, th][:, np.newaxis], axis=0)
                        update[:, :, th] = np.outer(pi[:, th], t)
                    update = np.apply_along_axis(self._avg_prob, 0, update)
                    # if d == 7 and s == 0:
                    #     print(update)
                    #     print(update.shape)
                    #     exit()

                    p_a_vector = []
                    p_a_vector_nums = []
                    for a_h in range(self.a_h):
                        tmp_p_a_vector = np.empty((0, self.th_h))
                        for ns, _p in self.ns[s][a_r][a_h]:
                            tmp_p_a_vector = np.concatenate(
                                [tmp_p_a_vector,
                                 self.a_vector[ns][th_r] * update[a_h, ns] +
                                 self.r[a_r, a_h, s, th_r, :] * pi[a_h, :]])
                        #     if d == 7 and a_r == 0 and s == 0:
                        #         print("-")
                        #         print(self.a_vector[ns][th_r], update[a_h, ns])
                        #         print(tmp_p_a_vector)
                        # if d == 7 and a_r == 0 and s == 0:
                        #     print("---")
                        #     print(tmp_p_a_vector)
                        p_a_vector.append(util.unique_for_raw(tmp_p_a_vector))
                        p_a_vector_nums.append(len(p_a_vector[-1]))
                    a_vector_a = np.zeros((np.prod(p_a_vector_nums), self.th_h))
                    # if d == 7 and a_r == 0 and s == 0:
                    #     print(p_a_vector)
                    #     exit()
                    for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                        a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                    # if d == 7 and a_r == 0 and s == 0:
                    #     print(a_vector_a)
                    #     exit()
                    a_vector_a = util.unique_for_raw(a_vector_a)
                    a_vector[s][th_r][a_r] = a_vector_a
        self.a_vector_a = {
            s: {th_r: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()} for
                th_r, vectorA in th_vector.items()} for s, th_vector in
            a_vector.items()} if bs is not None else a_vector
        # else:
        self.a_vector = {
            s: {th_r: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                th_r, vector in th_vector.items()}
            for s, th_vector in a_vector.items()} if bs is not None else a_vector
        # self.calc_belief()

    def calc_belief(self):
        self.beliefs = {}
        for th_r in range(self.th_r):
            beliefs = {0: np.array([0.5, 0.5])}
            s_candi = {0}
            while len(s_candi) > 0:
                s = s_candi.pop()
                for a_r in range(self.a_r):
                    for a_h in range(self.a_h):
                        for ns in np.where(self.t[a_r, a_h, s] > 0)[0]:
                            if ns == self.s - 1:
                                continue
                            if ns in beliefs:
                                print("error!", ns)
                                exit()
                            beliefs[ns] = beliefs[s] * self.h_pi[th_r][s][a_r][a_h]
                            beliefs[ns] /= np.sum(beliefs[ns])
            self.beliefs[th_r] = beliefs


    def value_a(self, s, th_r, a_r, b):
        return np.max(np.dot(self.a_vector_a[s][th_r][a_r], b))

    def value(self, s, th_r, b):
        return np.max(np.dot(self.a_vector[s][th_r], b))

    def get_best_action(self, s, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
