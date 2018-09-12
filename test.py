import numpy as np
import matplotlib.pyplot as plt
# from problem.tiger.pomdp_from import Tiger
# from problem.mod_tiger.pomdp_to import ModTiger
from problem.tiger.coop_pomdp_from import Tiger
from problem.correct.coop_pomdp_from import Correct

def make_belief(size = 2):
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    if size == 2:
        return np.concatenate(([b1], [b2]), axis=0).T
    b3 = np.zeros_like(b1)
    return np.concatenate(([b1], [b2], [b3]), axis=0).T

def test_tiger():
    b = make_belief(2)
    env = Tiger()
    # env= ModTiger()
    for d in [1, 2, 3, 6, 30]:
        env.calc_a_vector(d, b, with_a=False)
        # v = np.array([t.value(b[i]) for i in range(len(b))])
        v = np.array([env.value(0, b[i]) for i in range(len(b))])
        plt.plot(b[:, 0], v)
    plt.show()

def test_chef():
    b = make_belief(2)
    env = Correct(4, 2, [[2, 1, 1, 0], [1, 1, 0, 2]])
    for d in [3]:
        env.calc_a_vector(d, b, with_a=True)
        for a_r in range(env.a_r):
            v = np.array([env.value_a(0, a_r, b[i]) for i in range(len(b))])
            plt.plot(b[:, 0], v, label=a_r)
        plt.legend()
    plt.show()

if __name__ == '__main__':
    # test_tiger()
    test_chef()

