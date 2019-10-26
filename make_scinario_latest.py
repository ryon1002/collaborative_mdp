import itertools
import matplotlib.pyplot as plt
import importlib
import numpy as np

from algo.double_coop_irl import CoopIRL
from problem.ct.ct_data_mdp import ColorTrails
# from problem.ct.data1 import CTData
from problem.grid_graph.map import ItemMap
from problem.p_e.maze import Maze


def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T

if __name__ == '__main__':
    algo, target, main_th_r = 1, 0, 0
    # algo, target, main_th_r = 2, 1, 0
    index = 1
    # actions = [(0, 0), (0, 0), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (1, 2)]
    # actions = [(0, 2)] * 2 + [(0, 0)] * 2 + [(2, 0)] * 2 + [(2, 1)] * 2 + [(2, 1)]
    actions = [(0, 2)] * 2 + [(0, 0)] * 2 + [(2, 0)] * 2 + [(2, 1)] * 4 + [(1, 2)] * 3
    maze = Maze("problem/p_e/map_data/map1")
    # maze.show_world()
    step = 0
    # while True:
    #     h_a = int(input())
    #     print(h_a)
    # for a in actions:
    #     maze._move(*a)
    # maze.show_world()

    # # exit()
    # # env_module = importlib.import_module(f"problem.grid_graph.data{index}")
    # # print()
    #
    # for i in range(100000):
    #     if i % 1000 == 0:
    #         print(i)
    #     data = env_module.GraphData()
    #     data.check_data()
    #
    # exit()

    env = ColorTrails(env_module.CTData())
    env.make_data()
    irl = CoopIRL()

    # env.a_vector_a, env.h_pi = pickle.load(open("policy.pkl", "rb"))
    # env.make_scinario(0, 1, algo)
    # scinario = worst2.make_worst(2, env)
    # exit()

    b = make_belief()
    ii = 0
    if ii == 0:
        beliefs = {}
        for th_r in range(env.th_r):
            beliefs[th_r] = {}
            for s in range(env.s):
                beliefs[th_r][s] = np.array([0.5, 0.5])
    else:
        beliefs = env.calc_belief()
    for d in [7]:
        # env.calc_a_vector(d, beliefs, 1)
        irl.calc_a_vector(env, d, b, algo)

    # env.make_scinario(main_th_r, index, algo, target)
    # scinario = worst2.make_worst(index, 0, env)
    # pickle.dump((env.a_vector_a, env.h_pi), open("policy.pkl", "wb"))


    for a_r in range(env.a_r):
        v = np.array([irl.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        if np.max(v) < -999:
            continue
        plt.plot(b[:, 0], v, label=a_r)
        plt.legend()
        print(v)
    # plt.show()

    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(11, 0, a_r, b[i]) for i in range(len(b))])
    #     if np.max(v) < -999:
    #         continue
    #     plt.plot(b[:, 0], v, label=a_r)
    #     plt.legend()
    # plt.show()
    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(42, 0, a_r, b[i]) for i in range(len(b))])
    #     if np.max(v) < -999:
    #         continue
    #     plt.plot(b[:, 0], v, label=a_r)
    #     plt.legend()
    # plt.show()
