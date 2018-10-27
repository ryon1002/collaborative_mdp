import pickle
import numpy as np
import matplotlib.pyplot as plt
from problem.build.double_coop_irl_from2 import Build
# from problem.build.double_coop_irl_from import Build
from problem.build.data import BuildData


def make_belief(size=2):
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    if size == 2:
        return np.concatenate(([b1], [b2]), axis=0).T
    b3 = np.zeros_like(b1)
    return np.concatenate(([b1], [b2], [b3]), axis=0).T



def run_chef():
    b = make_belief(2)

    env = Build(BuildData())

    import time
    start = time.time()
    for d in [20]:
        env.calc_a_vector(d, b, with_a=True)
        # env.calc_a_vector(0, d, b, with_a=True)
    print(time.time() - start)
    # exit()
    for a_r in range(env.a_r - 1):
        # v = np.array([env.value_a(0, a_r, b[i]) for i in range(len(b))])
        v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        print(v)
    #     plt.plot(b[:, 0], v, label=a_r)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    run_chef()
