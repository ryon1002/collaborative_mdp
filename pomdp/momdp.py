import numpy as np
import itertools


class MOMDP(object):
    def __init__(self, x, y, z, a):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.tx = np.zeros((self.a, self.z, self.x, self.x))
        self.ty = np.zeros((self.a, self.x, self.y, self.y))
        self.r = np.zeros((self.y, self.a, self.x))
        self.o = np.zeros((self.y, self.a, self.x, self.z)) # start point

    def pre_calc(self):
        #p(y | x, a, z) \propto p(z | a, x, y) * p(y | a, x)
        # and multiply p(z | a, x, y) in advance
        self.update = np.zeros((self.a, self.x, self.z, self.y))

        for a in range(self.a):
            for x in range(self.x):
                for z in range(self.z):
                    self.update[a, x, z] = self.o[:, a, x, z]
                    # if np.sum(self.update[a, x, z]):
                    #     self.update[a, x, z] /= np.sum(self.update[a, x, z])
                    self.update[a, x, z] *= self.o[:, a, x, z]

        # pre-calc nx a->x->(nx, prob)
        self.nx = {x:{a:{z:self._ex_all_nx(x, a, z) for z in range(self.z)} for a in range(self.a)} for x in range(self.x)}

    def _ex_all_nx(self, x, a, z):
        arr = self.tx[a, z, x]
        idx = np.where(arr > 0)[0]
        return [i for i in zip(idx, arr[idx])]

    def calc_a_vector(self, d=1, bs=None, with_a=True):
        if d == 1:
            self.a_vector = {x: self.r[:, :, x].copy().T for x in range(self.x)}
            return
        self.calc_a_vector(d - 1, bs, False)
        a_vector = {}
        for x in range(self.x):
            a_vector[x] = {}
            for a in range(self.a):
                p_a_vector = []
                p_a_vector_nums = []
                for z in range(self.z):
                    for nx, p in self.nx[x][a][z]:
                        p_a_vector.append(self.a_vector[nx] * self.update[a, x, z] * p)
                        p_a_vector_nums.append(len(p_a_vector[-1]))

                a_vector_xa = np.zeros((np.prod(p_a_vector_nums), self.y))
                for m, i in enumerate(itertools.product(*[range(l) for l in p_a_vector_nums])):
                    a_vector_xa[m] = np.sum([p_a_vector[n][j] for n, j in enumerate(i)], axis=0)
                a_vector_xa = self.unique_for_raw(a_vector_xa)
                a_vector[x][a] = self.r[:, a, x].T + a_vector_xa
        if with_a:
            self.a_vector_a = {x: {a: self.prune(vector, bs) for a, vector in vectorA.items()} for x, vectorA in
                               a_vector.items()} if bs is not None else a_vector
        else:
            self.a_vector = {x: self.prune(np.concatenate(list(vector.values()), axis=0), bs) for x, vector in
                             a_vector.items()} if bs is not None else a_vector

    @staticmethod
    def unique_for_raw(a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))) \
            .view(a.dtype).reshape(-1, a.shape[1])

    @staticmethod
    def prune(a_vector, bs):
        index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
        return a_vector[index]

    def value_a(self, x, a, b):
        return np.max(np.dot(self.a_vector_a[x][a], b))

    def value(self, x, b):
        return np.max(np.dot(self.a_vector[x], b))

    def get_best_action(self, x, b):
        value_map = {k: np.max(np.dot(v, b)) for k, v in self.a_vector_a[x].viewitems()}
        return sorted(value_map.viewitems(), key=lambda a: a[1])[-1][0]

class Tiger(MOMDP):
    def __init__(self):
        super().__init__(2, 2, 2, 3)
        self.tx[:-1, :, :, 1] = 1
        self.tx[-1, :] = np.identity(2)

        self.ty[:, :] = np.identity(2)

        self.r[0, :2, 0] = [10, -100]
        self.r[1, :2, 0] = [-100, 10]
        self.r[:, -1, :] = [-1, 0]

        self.o[:, :2, :, :] = 0.5
        self.o[:, 2, 1, :] = 0.5
        self.o[0, 2, 0, :] = [0.8, 0.2]
        self.o[1, 2, 0, :] = [0.2, 0.8]
        print(self.o)

        self.pre_calc()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    b = np.concatenate(([b1], [b2]), axis=0).T
    t = Tiger()
    for d in [1, 2, 3, 6, 30]:
        t.calc_a_vector(d, b, with_a=False)
        v = np.array([t.value(0, b[i]) for i in range(len(b))])
        plt.plot(b[:, 0], v)
    plt.show()

