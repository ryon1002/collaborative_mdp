import numpy as np
import itertools


class Tiger(object):
    def __init__(self, depth=1, bs=None):
        self.s = 3
        self.a = 4
        self.z = 2
        self.t = np.zeros((self.a, self.s, self.s))  # P(s' | s, a)
        self.r = np.zeros((self.a, self.s))  # R(s, a)
        self.o = np.zeros((self.a, self.s, self.z))  # P(z | s, a)

        self.t[:2, :, -1] = 1
        self.t[-2:] = np.identity(3)
        self.t[-1:] = np.identity(3)

        self.r[:] = -100
        for i in range(2):
            self.r[i, i] = 10
        self.r[-1, :] = -2
        self.r[-2, :] = -1
        self.r[:, -1] = 0

        self.o[:, -1, :] = 0.5
        for i in range(2):
            self.o[:2, i, i] = 1
            self.o[-2, i, i] = 0.6
            self.o[-2, i, 1 - i] = 0.4
            self.o[-1, i, i] = 0.85
            self.o[-1, i, 1 - i] = 0.15
        self.orange = np.arange(self.z)
        self.pre_calc()
        self.calc_a_vector(depth, bs)

    def pre_calc(self):
        #P(s' | z, a) = \sum_s  { P(s' | s, a) * P(s | z, a) }
        # + * p(z | s, a)
        # self.next_s = np.zeros((self.a, self.z, self.s))
        self.next_s = np.zeros((self.a, self.z, self.s, self.s))
        for a in range(self.a):
            for z in range(self.z):
                z_sa = self.o[a, :, z]
                z_sa = z_sa / np.sum(z_sa)
                # print(self.t[a])
                # print(z_sa)
                # print(self.t[a] * z_sa[:, np.newaxis])
                self.next_s[a, z], self.t[a] * z_sa[:, np.newaxis]

                # self.next_s[a, z] = np.dot(self.t[a].T, z_sa)# * self.o[a, :, z]

                # self.next_s[a, z] = np.dot(self.t[a].T, z_sa)# * self.o[a, :, z]
        # exit()

    def calc_a_vector(self, d=1, bs=None):
        if d == 1:
            self.a_vector = self.r.copy()
            return
        self.calc_a_vector(d - 1, bs)
        a_vector = np.zeros((0, self.s))
        for a in range(self.a):
            # make partial a vector for each z
            ai_list = np.zeros((self.z, self.a_vector.shape[0], self.a_vector.shape[1]))
            for z in range(self.z):
                # ai_list[z] = self.a_vector * self.next_s[a, z] * self.o[a, :, z]
                print()
                exit()
                ai_list[z] = np.dot(self.next_s[a, z] * self.a_vector.T)# * self.o[a, :, z]
            # expand a_vector
            ai_list2 = np.zeros((len(self.a_vector) ** self.z, self.a_vector.shape[1]))
            for m, i in enumerate(itertools.product(range(len(self.a_vector)), repeat=self.z)):
                ai_list2[m] = np.sum(ai_list[self.orange, i], axis=0)
            ai_list2 = self.unique_for_raw(ai_list2)
            a_vector = np.vstack((a_vector, self.r[a] + ai_list2))
        self.a_vector = self.prune(a_vector, bs) if bs is not None else a_vector

    @staticmethod
    def unique_for_raw(a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(
            a.dtype).reshape(-1,
                             a.shape[1])

    def value(self, b):
        return np.max(np.dot(self.a_vector, b))

    @staticmethod
    def prune(a_vector, bs):
        index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
        return a_vector[index]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1

    b = np.concatenate(([b1], [b2], [np.zeros((len(b1)))]), axis=0).T

    env = Tiger(1, b)
    v = np.array([env.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)

    env2 = Tiger(2, b)
    v = np.array([env2.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)

    env3 = Tiger(3, b)
    v = np.array([env3.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)

    import datetime

    start = datetime.datetime.now()
    env4 = Tiger(30, b)
    v = np.array([env4.value(b[i]) for i in range(len(b))])
    print(datetime.datetime.now() - start)
    plt.plot(b[:, 0], v)

    plt.show()
