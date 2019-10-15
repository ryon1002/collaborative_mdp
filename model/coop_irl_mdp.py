import numpy as np


class CoopIRLMDP(object):
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

    # def func(self, arr):
    #     ret = np.zeros_like(arr)
    #     ret[np.argmax(arr)] = 1
    #     return ret

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


    # def value_a(self, s, th_r, a_r, b):
    #     return np.max(np.dot(self.a_vector_a[s][th_r][a_r], b))
    #
    # def value(self, s, th_r, b):
    #     return np.max(np.dot(self.a_vector[s][th_r], b))
    #
    # def get_best_action(self, s, b):
    #     value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[s].viewitems()}
    #     return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]
