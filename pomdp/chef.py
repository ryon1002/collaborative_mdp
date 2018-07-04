import numpy as np
import itertools


class Chef(object):
    def __init__(self, depth=1, bs=None):
        self.material_kind = 2
        self.recipe_kind = 2
        # self.max_material_num = 3
        self.max_material_num = 2

        self.materials_shape = tuple([self.max_material_num + 1] * self.material_kind)
        self.materials_size = np.prod(self.materials_shape)
        self.s = self.recipe_kind * self.materials_size + 1

        self.a = self.material_kind + 1
        self.z = 3
        self.t = np.zeros((self.a, self.s, self.s))  # P(s' | s, a)
        self.r = np.zeros((self.a, self.s))  # R(s, a)
        self.o = np.zeros((self.a, self.s, self.z))  # P(z | s, a)

        trans = np.zeros((self.material_kind, self.materials_size, self.materials_size))
        self.make_t(trans, (0, 0), self.max_material_num)

        goal_list = [(i, self.max_material_num - i) for i in range(self.max_material_num + 1)]
        for r in range(self.recipe_kind):
            self.t[:-1, r * self.materials_size: (r + 1) * self.materials_size,
            r * self.materials_size: (r + 1) * self.materials_size] = trans
            for g in goal_list:
                self.t[:-1, self.conv_s(g, r), -1] = 1
        self.t[-1] = np.identity(self.s)
        self.t[:, -1, -1] = 1

        # self.racipe = ((0, (3, 0)), (1, (0, 3)))
        self.racipe = ((0, (2, 0)), (1, (0, 2)))
        self.r[-1, :] = -1
        self.r[-1, -1] = 0
        for r_id, r_m in self.racipe:
            r_s = self.conv_s(r_m, r_id)
            prev_sa = np.where(self.t[:-1, :, r_s] == 1)
            prev_s, prev_a = prev_sa[1][0], prev_sa[0][0]
            self.r[prev_a, prev_s] = 10

        self.o[:, -1, -1] = 1
        self.o[:2, :, -1] = 1
        for r in range(self.recipe_kind):
            self.o[-1, r * self.materials_size: (r + 1) * self.materials_size, r] = 0.8
            self.o[-1, r * self.materials_size: (r + 1) * self.materials_size, 1 - r] = 0.2
        self.zrange = np.arange(self.z)
        self.pre_calc()
        print(self.r)
        # self.calc_a_vector(depth, bs)

    def conv_s(self, materials, recipe):
        return np.ravel_multi_index(materials, self.materials_shape) + recipe * self.materials_size

    def make_t(self, t, s_m, d):
        if d == 0:
            return
        for m in range(self.material_kind):
            next_s_m = list(s_m)
            next_s_m[m] += 1
            t[m, self.conv_s(s_m, 0), self.conv_s(next_s_m, 0)] = 1
            self.make_t(t, next_s_m, d - 1)

    def pre_calc(self):
        #P(s' | z, a) = \sum_s  { P(s' | s, a) * P(s | z, a) }
        # + * p(z | s, a)
        self.next_s = np.zeros((self.a, self.z, self.s))
        for a in range(self.a):
            for z in range(self.z):
                z_sa = self.o[a, :, z]
                div = np.sum(z_sa)
                if div == 0 :
                    div = 1
                z_sa = z_sa / div
                self.next_s[a, z] = np.dot(self.t[a].T, z_sa)# * self.o[a, :, z]

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
                ai_list[z] = self.a_vector * self.next_s[a, z]# * self.o[a, :, z]
                print(a,z)
                print(ai_list[z])
            # expand a_vector
            ai_list2 = np.zeros((len(self.a_vector) ** self.z, self.a_vector.shape[1]))
            for m, i in enumerate(itertools.product(range(len(self.a_vector)), repeat=self.z)):
                ai_list2[m] = np.sum(ai_list[self.zrange, i], axis=0)
            ai_list2 = self.unique_for_raw(ai_list2)
            print(a)
            print(ai_list2)
            a_vector = np.vstack((a_vector, self.r[a] + ai_list2))
        self.a_vector = self.prune(a_vector, bs) if bs is not None else a_vector

    @staticmethod
    def unique_for_raw(a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(
            a.dtype).reshape(-1, a.shape[1])

    def value(self, b):
        # print(self.a_vector)
        return np.max(np.dot(self.a_vector, b))

    @staticmethod
    def prune(a_vector, bs):
        index = np.unique(np.argmax(np.dot(a_vector, bs.T), axis=0))
        return a_vector[index]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)
    env = Chef()

    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1

    b = np.zeros((len(b1), env.s))
    b[:, env.conv_s((0,0), 0)] = b1
    b[:, env.conv_s((0,0), 1)] = b2

    env.calc_a_vector(2, b)
    exit()
    v = np.array([env.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)
    plt.show()
    exit()

    env.calc_a_vector(4, b)
    v = np.array([env.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)
    plt.show()
    exit()

    env3 = Tiger(6, b)
    v = np.array([env3.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)

    import datetime

    start = datetime.datetime.now()
    env4 = Tiger(30, b)
    v = np.array([env4.value(b[i]) for i in range(len(b))])
    print(datetime.datetime.now() - start)
    plt.plot(b[:, 0], v)

    plt.show()
