import json
import os
import numpy as np
import worst2
import matplotlib.pyplot as plt

from problem.ct.double_coop_irl_from2 import ColorTrails
from problem.graph import train1, train2, data01, data01_2, data01_0_1,\
    data02, data02_2, data03, data03_2, data04, data04_2
import make_graph
# from problem.ct.data1 import CTData
# from problem.ct.data2 import CTData
# from problem.ct.data3 import CTData
# from problem.ct.data4 import CTData
# from problem.ct.data10 import CTData
from problem.ct.data11 import CTData

def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T


# def run_ct(graph_id, dir_name, algo, obj, pref):
    # os.makedirs(dir_name, exist_ok=True)
    # b = make_belief()
    # env = Graph(graph)
    # json_data = make_graph.make_json(graph, algo, obj)
    # json.dump(json_data, open(dir_name + "data.json", "w"), indent=4)
    # for ii in range(5):
    #     if algo == 3:
    #         scinario = worst.make_worst(pref, graph)
    #     else:
    #         if ii == 0:
    #             beliefs = {}
    #             for th_r in range(env.th_r):
    #                 beliefs[th_r] = {}
    #                 for s in range(env.s):
    #                     beliefs[th_r][s] = np.array([0.5, 0.5])
    #         else:
    #             beliefs = env.calc_belief()
    #         for d in [7]:
    #             env.calc_a_vector(d, beliefs, algo)
    #         print(beliefs)
    #         scinario = env.make_scinario(pref)
    #     json.dump(scinario, open(dir_name + "scinario.json", "w"), indent=4)

    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
    #     print(ii)
    #     print(v)
    #     plt.plot(b[:, 0], v, label=a_r)
    #     # plt.legend()
    #     plt.show()

import pickle
if __name__ == '__main__':
    algo, target, main_th_r = 1, 0, 0
    # algo, target, main_th_r = 2, 1, 0
    index = 11
    env = ColorTrails(CTData())
    env.make_data(index)
    # exit()

    # env.a_vector_a, env.h_pi = pickle.load(open("policy.pkl", "rb"))
    # env.make_scinario(0, 1, algo)
    # scinario = worst2.make_worst(2, env)
    # exit()

    # exit()
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
        env.calc_a_vector(d, b, algo)
    env.make_scinario(main_th_r, index, algo, target)
    scinario = worst2.make_worst(index, 0, env)
    # pickle.dump((env.a_vector_a, env.h_pi), open("policy.pkl", "wb"))


    for a_r in range(env.a_r):
        v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        if np.max(v) < -999:
            continue
        plt.plot(b[:, 0], v, label=a_r)
        plt.legend()
    plt.show()
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
