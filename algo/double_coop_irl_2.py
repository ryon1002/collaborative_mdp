import numpy as np
import itertools
import pickle
import os
from . import util
from . import prob_util


class CoopIRL(object):
    def __init__(self):
        # self.default_belief = np.array([0.5, 0.5])
        self.default_belief = np.array([1.0])

    def calc_a_vector(self, env, d, bs=None, algo=1, use_dump=False, save_dump=False):
        if env.th_h == 2:
            self.default_belief = np.array([0.5, 0.5])
        elif env.th_h == 1:
            self.default_belief = np.array([1.0])
        if d == 1:
            self.a_vector = {}
            for s in range(env.s):
                self.a_vector[s] = {}
                for th_r in range(env.th_r):
                    self.a_vector[s][th_r] = np.zeros((1, env.th_h))
            self.beliefs = {}
            for th_r in range(env.th_r):
                self.beliefs[th_r] = {}
                for s in range(env.s):
                    self.beliefs[th_r][s] = self.default_belief
                    # self.beliefs[th_r][s] = np.array([0.5, 0.5])
            return
        if use_dump and os.path.exists(f"store/{d}.pkl"):
            self.a_vector, self.a_vector_a, self.beliefs, self.h_pi = \
                pickle.load(open(f"store/{d}.pkl", "rb"))
            return
        else:
            self.calc_a_vector(env, d - 1, bs, algo, use_dump, save_dump)

        a_vector = {s: {th_r: {} for th_r in range(env.th_r)} for s in range(env.s)}

        # inv_r_pi = {}
        # for s in range(env.s):
        #     r_val = np.zeros((env.a_r, env.th_r))
        #     for th_r in range(env.th_r):
        #         for a_r in range(env.a_r):
        #             tmp_th_h_val = np.zeros(env.th_h)
        #             for th_h in range(env.th_h):
        #                 tmp_a_h_val = np.zeros(env.a_h)
        #                 for a_h in range(env.a_h):
        #                     for ns, _p in env.ns[s][a_r][a_h]:
        #                         tmp_a_h_val[a_h] += np.max(self.a_vector[ns][th_r][:, th_h])
        #                     tmp_a_h_val[a_h] += env.r[a_r, a_h, s, th_r, th_h]
        #                 tmp_th_h_val[th_h] = np.max(tmp_a_h_val)
        #
        #             # r_val[a_r, th_r] = np.mean(tmp_th_h_val)
        #             belief = self.beliefs[th_r][s] if s in self.beliefs[th_r] else self.default_belief
        #             r_val[a_r, th_r] = np.dot(tmp_th_h_val, belief)
        # if algo == 1:
        #     r_pi = np.apply_along_axis(prob_util._exp_q_prob, 0, r_val / 10)  # full bayes
        #     # r_pi = np.apply_along_axis(self._max_q_prob, 0, r_val)  # full bayes
        #     # inv_r_pi[s] = np.apply_along_axis(self._max_q_prob, 1, r_pi)
        #     inv_r_pi[s] = np.apply_along_axis(prob_util._exp_q_prob, 1, r_pi)
        # if d == 6 and s == 0:
        #     print(r_pi)
        #     print(r_val)
        #     print(inv_r_pi[0])
        #     exit()
        # elif algo == 0:
        #     r_pi = np.apply_along_axis(prob_util._max_q_prob, 1, r_val) # bias
        #     inv_r_pi[s] = np.apply_along_axis(prob_util._max_q_prob, 1, r_pi)
        # if d == 6 and s == 0:
        #     print(r_pi)
        #     print(r_val)
        #     print(inv_r_pi[0])
        #     exit()
        # print(inv_r_pi.shape())
        if algo == 1:
            inv_r_pi = self.h_belief.copy()
        # print(inv_r_pi.shape)
        # exit()
        self.h_pi = {}
        for th_r in range(env.th_r):
            if algo == 2:
                inv_r_pi = np.zeros((env.th_h, env.s, env.th_r))
                inv_r_pi[:, :, th_r] = 1
            self.h_pi[th_r] = {}
            for s in range(env.s):
                # if algo == 2:
                #     inv_r_pi[s] = np.zeros((env.a_r, env.th_r))
                #     inv_r_pi[s][:, th_r] = 1
                self.h_pi[th_r][s] = {}
                for a_r in range(env.a_r):
                    q_vector_2 = np.zeros((env.a_h, env.th_h))
                    for a_h in range(env.a_h):
                        for th_r2 in range(env.th_r):
                            q_vector2_a = np.zeros((0, env.th_h))
                            for ns, p in env.ns[s][a_r][a_h]:
                                q_vector2_a = np.concatenate([q_vector2_a,
                                                              env.r[a_r, a_h, s, th_r2] * p +
                                                              self.a_vector[ns][th_r2] * p *
                                                              inv_r_pi[:, ns, th_r2]])
                                # q_vector2_a += ((env.r[a_r, a_h, s, th_r2] +
                                #                  self.a_vector[ns][th_r2]) * p *
                                #                 inv_r_pi[:, ns, th_r2])
                                # q_vector2_a += ((self.a_vector[ns][th_r2]) * p *
                                #                 inv_r_pi[:, ns, th_r2])
                            q_vector_2[a_h] += np.max(q_vector2_a, axis=0)# +env.r[a_r, a_h, s, th_r2]
                            # q_vector_2[a_h, th_r2] = np.max(np.sum(q_vector2_a, axis=0))

                        # for th_r2 in range(env.th_r):
                        #     q_vector2_a = np.zeros((0, env.th_h))
                        #     for ns, _p in env.ns[s][a_r][a_h]:
                        #         q_vector2_a = np.concatenate([q_vector2_a,
                        #                                       env.r[a_r, a_h, s, th_r2] +
                        #                                       self.a_vector[ns][th_r2]])
                        #     if d == 6 and s == 0 and a_r == 0:
                        #         print(a_h, ns, th_r2, self.a_vector[ns][th_r2], self.r[a_r, a_h, s, th_r2])
                        # if d == 6 and s == 0 and a_r == 0:
                        #     print(a_h, q_vector2_a)
                        #     print(inv_r_pi[s][a_r, th_r2])
                        #### q_vector_2[a_h] += np.max(q_vector2_a * inv_r_pi[s][a_r, th_r2], axis=0)
                        # q_vector_2[a_h] += np.max(q_vector2_a * inv_r_pi[s, th_r2], axis=0)

                    # if env.th == 2:
                    #     pass
                    # else:
                    pi = np.apply_along_axis(prob_util._max_q_prob, 1, q_vector_2)
                    pi = np.apply_along_axis(prob_util._max_q_prob, 0, pi)
                    # pi = np.apply_along_axis(prob_util._exp_q_prob, 0, q_vector_2, 0.1)
                    # if d == 6 and s == 0 and a_r == 0:
                    #     print(q_vector_2)
                    #     print(pi)
                    #     exit()

                    #### self.h_pi[th_r][s][a_r] = np.apply_along_axis(prob_util._exp_q_prob, 1, pi, 0.1)

                    # self.h_pi[th_r][s][a_r] = np.apply_along_axis(self._exp_q_prob, 1,
                    #                                               q_vector_2 / 1000)
                    # if d == 6 and s == 53 and a_r == 0:
                    #     print(self.h_pi)

                    update = np.empty((env.a_h, env.s, env.th_h))
                    for th in range(env.th_h):
                        t = np.sum(env.t[a_r, :, s] * pi[:, th][:, np.newaxis], axis=0)
                        update[:, :, th] = np.outer(pi[:, th], t)
                    update = np.apply_along_axis(prob_util._avg_prob, 0, update)

                    p_a_vector = []
                    p_a_vector_nums = []
                    for a_h in range(env.a_h):
                        tmp_p_a_vector = np.empty((0, env.th_h))
                        for ns, _p in env.ns[s][a_r][a_h]:
                            tmp_p_a_vector = np.concatenate(
                                [tmp_p_a_vector,
                                 self.a_vector[ns][th_r] * update[a_h, ns] +
                                 env.r[a_r, a_h, s, th_r, :] * pi[a_h, :]])
                        p_a_vector.append(util.unique_for_raw(tmp_p_a_vector))
                        p_a_vector_nums.append(len(p_a_vector[-1]))
                    a_vector_a = np.zeros((np.prod(p_a_vector_nums), env.th_h))
                    for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                        a_vector_a[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                    # if a_r == 1 and d == 8 and s == 8:
                    #     print(a_vector_a)
                    #     exit()
                    a_vector_a = util.unique_for_raw(a_vector_a)
                    a_vector[s][th_r][a_r] = a_vector_a
        self.a_vector_a = {
            s: {th_r: {a_r: util.prune(vector, bs) for a_r, vector in vectorA.items()} for
                th_r, vectorA in th_vector.items()} for s, th_vector in
            a_vector.items()} if bs is not None else a_vector

        self.a_vector = {
            s: {th_r: util.prune(np.concatenate(list(vector.values()), axis=0), bs) for
                th_r, vector in th_vector.items()}
            for s, th_vector in a_vector.items()} if bs is not None else a_vector

        if save_dump:
            pickle.dump((self.a_vector, self.a_vector_a, self.beliefs, self.h_pi),
                        open(f"store/{d}.pkl", "wb"))

        print(d)
        # if d == 7:
        #     print(self.a_vector_a[1])
        #     print(self.a_vector[1])
        # #     print(self.ns[8])
        #     print(self.a_vector_a[0][0])
        #     print(self.h_pi[0][0][0])
        #     print(self.h_pi[0][0][2])
        #     print(self.a_vector[0][0])
        #     print(self.a_vector[1][0])
        #     print(self.a_vector[53][0])
        #     print(self.a_vector[105][0])
        #     print(self.a_vector[134][0])
        #     exit()
        #     print(self.a_vector_a[12])
        # #     print(self.a_vector_a[12])
        # #     print(self.a_vector[12])
        # #     # print(self.ns[12])
        #     exit()
        # self.calc_belief()

    # def calc_belief(self, env):
    #     self.beliefs = {}
    #     for th_r in range(env.th_r):
    #         beliefs = {0: np.array([0.5, 0.5])}
    #         s_candi = {0}
    #         while len(s_candi) > 0:
    #             s = s_candi.pop()
    #             for a_r in range(env.a_r):
    #                 for a_h in range(env.a_h):
    #                     for ns in np.where(env.t[a_r, a_h, s] > 0)[0]:
    #                         if ns == env.s - 1:
    #                             continue
    #                         if ns in beliefs:
    #                             print("error!", ns)
    #                             exit()
    #                         beliefs[ns] = beliefs[s] * self.h_pi[th_r][s][a_r][a_h]
    #                         beliefs[ns] /= np.sum(beliefs[ns])
    #         self.beliefs[th_r] = beliefs

    def calc_h_belief(self, env, q, beta=1):
        self.h_belief = np.empty((env.th_h, env.s, env.th_r))
        self.h_belief[:, :, :] = 0.5
        for th_h in range(env.th_h):
            prob = np.apply_along_axis(prob_util._exp_q_prob, 0, q[:, th_h], beta)
            s_candi = {0}
            while len(s_candi) > 0:
                s = s_candi.pop()
                for a_r in range(env.a_r):
                    for a_h in range(env.a_h):
                        for ns in np.where(env.t[a_r, a_h, s] > 0)[0]:
                            if ns == env.s - 1:
                                continue
                            self.h_belief[th_h, ns] = self.h_belief[th_h, s] * prob[:, a_r, s]
                            self.h_belief[th_h, ns] /= np.sum(self.h_belief[th_h, ns])
                            s_candi.add(ns)

    def value_a(self, s, th_r, a_r, b):
        return np.max(np.dot(self.a_vector_a[s][th_r][a_r], b))

    def value(self, s, th_r, b):
        return np.max(np.dot(self.a_vector[s][th_r], b))

    def get_best_action(self, s, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
