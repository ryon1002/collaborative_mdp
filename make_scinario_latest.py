import itertools
import matplotlib.pyplot as plt
import importlib
import numpy as np

from algo.double_coop_irl import CoopIRL
from problem.ct.ct_data_mdp import ColorTrails
# from problem.ct.data1 import CTData
from problem.grid_graph.map import ItemMap


def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T

if __name__ == '__main__':
    algo, target, main_th_r = 1, 0, 0
    # algo, target, main_th_r = 2, 1, 0
    index = 1
    env_module = importlib.import_module(f"problem.grid_graph.data{index}")
    im = ItemMap()
    item_dist, agent_dist = im.make_matrix()
    data = env_module.GraphData(item_dist, agent_dist)
    check = data.check_data()
    exit()
    dist_base = np.zeros((6, 2), dtype=int)
    dist_base[:, 0] = [-3, -2, -1, 1, 2, 3]
    # dist_base[:, 0] = [-6, -4, -2, 2, 4, 6]
    min = -1
    for l in itertools.product(range(-6, 7), repeat=6):
        dist_base[:, 1] = l
        im = ItemMap(dist_base)
        item_dist, agent_dist = im.make_matrix()
        data = env_module.GraphData(item_dist, agent_dist)
        check = data.check_data()
        if not check:
            t_min = np.min(np.abs(im.items[1:, 1] - im.items[:-1, 1]))
            if t_min > min:
                print(t_min)
                print(item_dist, "\n",agent_dist)
                print(im.items)
                print()
                min = t_min
            # exit()
    exit()

    for i in range(100000):
        if i % 1000 == 0:
            print(i)
        data = env_module.GraphData()
        data.check_data()

    exit()
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
