import pickle
import numpy as np
import matplotlib.pyplot as plt
# from problem.tiger.pomdp_from import Tiger
# from problem.mod_tiger.pomdp_to import ModTiger
from problem.tiger.coop_pomdp_from import Tiger
# from problem.correct.coop_pomdp_from import Correct
# from problem.correct.coop_irl_from import Correct
# from problem.correct.double_coop_irl_from import Correct
# from problem.correct.double_coop_irl_from2 import Correct
from problem.correct.double_coop_irl_from2_2 import Correct


def make_belief(size=2):
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
    # print((1, 7) == (1, 7))
    # exit()
    # print("test")
    b = make_belief(2)
    # print(0b0101_11000)
    # print(0b0010_10000)
    # print(0b0010_10000)
    # exit()
    # env = Correct(4, 2, [[2, 1, 1, 0], [1, 1, 0, 2]])
    # env = Correct(4, 2, [[1, 0, 2, 1], [0, 1, 1, 2]])

    # env = Correct(11, 1,
    #               [[1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    #                [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    #                [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    #                [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]],
    #               [[-1, -1, -1, -1, -3, -1, -1, -1, -1, -1, -1],
    #                [-1, -1, -1, -1, -1, -3, -1, -1, -1, -1, -1]])

    # env = Correct(10, 1,
    #               [[[1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    #                 [0, 0, 1, 0, 1, 0, 1, 1, 1, 0]],
    #                [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #                 [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]]],
    #               [[-3, -3, -1, -1, -1, -1, -1, -1, -1, -1],
    #                [-1, -1, -1, -1, -3, -1, -1, -1, -1, -1]])

    # env = Correct(4, 5, 1,
    #               [[[1, 0, 1, 0, 1, 1, 0, 0, 0],
    #                 [0, 1, 1, 0, 1, 0, 1, 0, 0]],
    #                [[0, 1, 0, 1, 1, 0, 0, 1, 0],
    #                 [1, 0, 0, 1, 1, 0, 0, 0, 1]]],
    #               [[-3, -1, -1, -1, -1, -1, -1, -1, -1],
    #                [-1, -3, -1, -1, -1, -1, -1, -1, -1]])

    # env = Correct(4, 3, 1,
    #               [[[1, 0, 0, 1, 1, 1, 0],
    #                 [1, 0, 1, 0, 1, 0, 1]],
    #                [[0, 1, 1, 0, 1, 0, 1],
    #                 [0, 1, 0, 1, 1, 1, 0]]],
    #               [[-1, -1, -1, -3, -1, -1, -1],
    #                [-1, -1, -3, -1, -1, -1, -1]])
    # env = Correct(4, 3, 1,
    #               [[[1, 0, 0, 1, 1, 1, 0],
    #                 [1, 0, 1, 0, 1, 0, 1]],
    #                [[0, 1, 0, 1, 1, 0, 1],
    #                 [0, 1, 1, 0, 1, 1, 0]]],
    #               [[-1, -1, -1, -3, -1, -1, -1],
    #                [-1, -1, -3, -1, -1, -1, -1]])

    # env = Correct(4, 5, 1,
    #               [[[1, 0, 1, 0, 0, 1, 0, 1, 0],
    #                 [1, 0, 0, 1, 0, 1, 0, 0, 1],
    #                 [1, 0, 1, 0, 1, 0, 0, 1, 0],
    #                 [1, 0, 0, 1, 1, 0, 0, 0, 1]],
    #                [[0, 1, 1, 0, 0, 0, 1, 0, 1],
    #                 [0, 1, 0, 1, 0, 0, 1, 1, 0],
    #                 [0, 1, 1, 0, 1, 0, 0, 0, 1],
    #                 [0, 1, 0, 1, 1, 0, 0, 1, 0]]],
    #               [[-1, -1, -5, -1, -3, -1, -1, -1, -1],
    #                [-1, -1, -1, -5, -3, -1, -1, -1, -1]])
    # env = Correct(4, 5, 1,
    #               [[[1, 0, 1, 0, 0, 1, 0, 1, 0],
    #                 [1, 0, 0, 1, 0, 1, 0, 0, 1],
    #                 [1, 0, 1, 0, 1, 0, 0, 1, 0],
    #                 [1, 0, 0, 1, 1, 0, 0, 0, 1]],
    #                [[0, 1, 1, 0, 0, 0, 1, 1, 0],
    #                 [0, 1, 0, 1, 0, 0, 1, 0, 1],
    #                 [0, 1, 1, 0, 1, 0, 0, 1, 0],
    #                 [0, 1, 0, 1, 1, 0, 0, 0, 1]]],
    #               [[-10, -10, -15, -10, -13, -10, -10, -10, -10],
    #                [-10, -10, -10, -15, -13, -10, -10, -10, -10]])
    # [[-1, -1, -5, -1, -3, -1, -1, -1, -1],
    #  [-1, -1, -1, -5, -3, -1, -1, -1, -1]])

    # env = Correct(6, 5, 1,
    #               [[[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    #                 [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    #                 [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    #                 [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    #                 [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    #                 [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    #                 [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    #                 [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1]],
    #                [[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    #                 [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    #                 [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    #                 [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    #                 [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    #                 [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    #                 [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    #                 [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]]],
    #               [[-20, -20, -10, -10, -30, -10, -15, -10, -10, -10, -10],
    #                [-20, -20, -10, -10, -10, -30, -15, -10, -10, -10, -10]])
    #               # [[-20, -20, -10, -10, -30, -10, -15, -10, -10, -10, -10]])
    #
    # pickle.dump(env, open("env.pkl", "wb"))
    env = pickle.load(open("env.pkl", "rb"))

    print("test")

    import time
    start = time.time()
    for d in [2]:
        env.calc_a_vector(d, b, with_a=True)
    # env.calc_a_vector(0, d, b, with_a=True)
    print(time.time() - start)
    for a_r in range(env.a_r):
        v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        print(v)
        plt.plot(b[:, 0], v, label=a_r)
    # break
    # exit()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_chef()
