import numpy as np

def do_value_iteration(t, r, gamma=1):
    v = np.zeros(t.shape[2])
    # print(r.shape)
    # # rr = rr
    # rr = np.zeros((r.shape[0], r.shape[1], r.shape[2], r.shape[2]))
    # for a in range(r.shape[0]):
    #     for b in range(r.shape[1]):
    #         for s in range(r.shape[2]):
    #             rr[a, b, s, :] = r[a, b, s]
    # # print(rr[0, 0])
    # print(r.shape)
    #
    # exit()
    for _i in range(500):
        n_v = np.sum(t * r + t * v * gamma, axis=2)
        n_v = np.max(n_v, axis=0)
        chk = np.sum(np.abs(n_v - v))
        if chk < 1e-6:
            break
        v = n_v
    q = np.sum(t * r + t * n_v * gamma, axis=2)
    return q

    # def doSoftValueItaration(self, w, itr=500):
    #     r = self.t * np.dot(self.saFeture, w).T[:, :, np.newaxis]
    #     v = np.zeros(self.t.shape[1])
    #     p = np.ones((self.s, self.a))
    #
    #     for _i in range(itr):
    #         q = np.sum(self.t * r + ((self.t * v).T * p).T * 0 , axis=2).T
    #         v = np.apply_along_axis(self.softmax, 1, q)
    #         p = np.exp(q - v[:, np.newaxis])
    #     self.q = q.T
    #     self.p = p
    #
    # def doSoftValueItaration2(self, w, itr=500):
    #     r = self.t * np.dot(self.saFeture, w).T[:, :, np.newaxis]
    #     v = np.zeros(self.t.shape[1])
    #     for _i in range(3):
    #         n_v = np.sum(self.t * r + self.t * v * self.d, axis=2)
    #         n_v = np.apply_along_axis(self.softmax, 0, n_v)
    #         print n_v[39]
    #         chk = np.sum(np.abs(n_v - v))
    #         if chk < 1e-6:
    #             break
    #         v = n_v
    #     self.q = np.sum(self.t * self.r + self.t * n_v * self.d, axis=2)
    #
