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
        # print("nn", ret, arr)
        # if ret[0] != ret[0]:
        #     exit()
        # if ret[1] != ret[1]:
        #     exit()
        # if len(ret) > 2:
        #     if ret[2] != ret[2]:
        #         exit()
        #     if ret[3] != ret[3]:
        #         exit()
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
        # print(self.ns[9][2])
        # exit()

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
                    # for a_r in range(self.a_r):
                    #     print(a_r, s, th_r)
                    #     print(np.max(self.r[a_r, :, s, th_r, :], axis=0))
                    #     print(self.a_vector[s][th_r])
                    #     print(self.a_vector[s][th_r].shape)
                    #     self.a_vector[s][th_r] = \
                    #         np.concatenate(self.a_vector[s][th_r],
                    #                        np.max(self.r[a_r, :, s, th_r, :], axis=0))

                    # for
                    # for th_h in range(self.th_h):
                    #     self.a_vector[s][th_r][0][th_h] = np.max(self.r[:, :, s, th_r, th_h])
                    # if s == 7:
                    #     for t_a_r in range(self.a_r):
                    #         print(np.max(self.r[t_a_r, :, s, th_r, :], axis=0))
                    #     exit()
            # print(self.a_vector[2])
            # for s in range(self.s):
            #     print(s, self.a_vector[s])
            # exit()
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
                                # if d == 6 and s == check_s:# and th_r == 1:
                                #     # print("test2", a_r, ns, a_h, th_r, self.a_vector[ns][th_r])
                                #     print("test2", a_r, ns, a_h, th_r, self.a_vector[ns][th_r])
                                tmp_a_h_val[a_h] += np.max(self.a_vector[ns][th_r][:, th_h])
                                # print("test2", ns, a_h, th_r, self.a_vector[ns][th_r][:, th_h])
                                # exit()
                            tmp_a_h_val[a_h] += self.r[a_r, a_h, s, th_r, th_h]
                        # if d == 6 and s == check_s:# and th_r == 1:
                        #     print("test", th_h, th_r, a_r, tmp_a_h_val)
                        tmp_th_h_val[th_h] = np.max(tmp_a_h_val)

                    # if d == 2 and s == 0:
                    #     print(tmp_th_h_val)

                    # if d == 6 and s == check_s:
                    #     print("test3", th_r, a_r, tmp_th_h_val)
                    r_val[a_r, th_r] = np.mean(tmp_th_h_val)
            if algo == 1:
                r_pi = np.apply_along_axis(self._max_q_prob, 0, r_val)  # full bayes
            elif algo == 0:
                r_pi = np.apply_along_axis(self._max_q_prob, 1, r_val) # bias
            # r_pi = np.apply_along_axis(self._exp_q_prob, 0, r_val)
            # r_pi = np.apply_along_axis(self._exp_q_prob, 1, r_val)
            # print(r_val, r_pi)
            if algo == 2:
                inv_r_pi[s] = np.zeros((self.a_r, self.th_r))
                for th_r in range(self.th_r):
                    inv_r_pi[s][:, th_r] = 1
            else:
                inv_r_pi[s] = np.apply_along_axis(self._max_q_prob, 1, r_pi)
            # if d == 6 and s == check_s:
            #     print(r_val)
            #     print(r_pi)
            #     print(inv_r_pi[s])
                # exit()
            # inv_r_pi[s] = np.apply_along_axis(self._exp_q_prob, 1, r_pi)

        self.h_pi = {}
        for th_r in range(self.th_r):
            self.h_pi[th_r] = {}
            for s in range(self.s):
                self.h_pi[th_r][s] = {}
                for a_r in range(self.a_r):
                    q_vector_2 = np.zeros((self.a_h, self.th_h))
                    for a_h in range(self.a_h):
                        for th_r2 in range(self.th_r):
                            q_vector2_a = np.zeros((0, self.th_h))
                            # print("check", s, a_r, a_h, self.ns[s][a_r][a_h])
                            for ns, _p in self.ns[s][a_r][a_h]:
                                q_vector2_a = np.concatenate([q_vector2_a,
                                                              self.r[a_r, a_h, s, th_r2] +
                                                              self.a_vector[ns][th_r2]])
                            if d == 6 and s == check_s and a_r == 2:
                                print(self.a_vector[ns][th_r2])
                                print(self.a_vector_a[ns][th_r2])
                                print("check", a_h, th_r2)
                                print(ns, _p, q_vector2_a)
                                print(inv_r_pi[s][a_r, th_r2])
                                # print(ns, _p, q_vector_2)
                            q_vector_2[a_h] += np.max(q_vector2_a * inv_r_pi[s][a_r, th_r2], axis=0)
                    pi = np.apply_along_axis(self._max_q_prob, 0, q_vector_2)
                    if s == check_s and a_r == 2 and d == 6:
                        print(q_vector_2)
                        print(pi)
                        # exit()
                    # self.h_pi[th_r][s][a_r] = np.apply_along_axis(self._exp_q_prob, 1, pi)
                    self.h_pi[th_r][s][a_r] = np.apply_along_axis(self._exp_q_prob, 1,
                                                                  q_vector_2 / 1000)
                    # if d == 3 and a_r == 1:
                    #     print(th_r, s, a_r)
                    #     print(q_vector_2)
                    #     print(pi)
                    #     print(self.h_pi[th_r][s][a_r])
                    #     exit()
                    # if d == 3 and s == 0 and a_r == 0:
                    #     print(th_r, q_vector_2)
                    #     print(th_r, pi)

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
                    # if d == 6 and s == check_s:
                    #     # print(a_r, p_a_vector)
                    #     print("r", th_r, a_r, self.r[a_r, a_h, s, th_r, :])
                    # print(pi)
                    a_vector_a = np.zeros((np.prod(p_a_vector_nums), self.th_h))
                    for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                        a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                    a_vector_a = util.unique_for_raw(a_vector_a)
                    # a_vector[s][a_r] = a_vector_a
                    # if d == 6 and s == check_s:
                    #     print(s, th_r, a_r, a_vector_a)
                    a_vector[s][th_r][a_r] = a_vector_a
        # if d == 5:
        #     print(a_vector[check_s])
        #     exit()

        # if d == 6:
        #     print(a_vector[check_s])
        #     print(a_vector[1][0])
        #     print(a_vector[4][0])
        #     print(a_vector[7][0])
        #     print(a_vector[10][0])
        #     exit()
        # if with_a:
        self.a_vector_a = {
            s: {th_r: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()} for
                th_r, vectorA in th_vector.items()} for s, th_vector in
            a_vector.items()} if bs is not None else a_vector
        # else:
        self.a_vector = {
            s: {th_r: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                th_r, vector in th_vector.items()}
            for s, th_vector in a_vector.items()} if bs is not None else a_vector
        if d == 6:
            print(check_s)
            print(self.a_vector[check_s])
            print(self.a_vector_a[check_s])
            # print(self.a_vector[4][0])
            # print(self.a_vector[7][0])
            # print(self.a_vector[10][0])
            # exit()

    def value_a(self, s, th_r, a_r, b):
        return np.max(np.dot(self.a_vector_a[s][th_r][a_r], b))

    def value(self, s, th_r, b):
        return np.max(np.dot(self.a_vector[s][th_r], b))

    def get_best_action(self, s, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
