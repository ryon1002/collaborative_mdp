import json
import os
import numpy as np
import worst
import matplotlib.pyplot as plt

from problem.graph.double_coop_irl_from2 import Graph
from problem.graph import train1, train2, data01, data01_2, data01_0_1,\
    data02, data02_2, data03, data03_2, data04, data04_2
import make_graph

def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T


def run_chef(graph_id, dir_name, algo, obj, pref):
    if graph_id == "t1" : graph = train1.GraphData()
    elif graph_id == "t2" : graph = train2.GraphData()
    elif graph_id == "1" : graph = data01.GraphData()
    elif graph_id == "1_2" : graph = data01_2.GraphData()
    elif graph_id == "1_3" : graph = data01_0_1.GraphData()
    elif graph_id == "2" : graph = data02.GraphData()
    elif graph_id == "2_2" : graph = data02_2.GraphData()
    elif graph_id == "3" : graph = data03.GraphData()
    elif graph_id == "3_2" : graph = data03_2.GraphData()
    elif graph_id == "4" : graph = data04.GraphData()
    elif graph_id == "4_2" : graph = data04_2.GraphData()

    dir_name = "scinario/" + dir_name + "/"
    os.makedirs(dir_name, exist_ok=True)
    b = make_belief()
    env = Graph(graph)
    json_data = make_graph.make_json(graph, algo, obj)
    json.dump(json_data, open(dir_name + "data.json", "w"), indent=4)
    for ii in range(1):
        if algo == 3:
            scinario = worst.make_worst(pref, graph)
        else:
            if ii == 0:
                beliefs = {}
                for th_r in range(env.th_r):
                    beliefs[th_r] = {}
                    for s in range(env.s):
                        beliefs[th_r][s] = np.array([0.5, 0.5])
            else:
                beliefs = env.calc_belief()
            for d in [7]:
                # env.calc_a_vector(d, beliefs, algo)
                env.calc_a_vector(d, b, algo)
            print(beliefs)
            scinario = env.make_scinario(pref)
        json.dump(scinario, open(dir_name + "scinario.json", "w"), indent=4)

    for a_r in range(env.a_r):
        v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        print(ii)
        print(v)
        if v[0] == -2000:
            continue
        plt.plot(b[:, 0], v, label=a_r)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # run_chef("t1", "train1", 0, 0, 0)
    # run_chef("t2", "train2", 0, 0, 1)
    # run_chef("1", "1", 0, 0, 0)
    # run_chef("1_3", "2", 1, 1, 1)
    # run_chef("1", "3", 2, 1, 0)
    # run_chef("1_2", "4", 3, 1, 1)
    run_chef("2", "5", 0, 0, 0)
    # run_chef("2_2", "6", 1, 0, 0)
    # run_chef("2", "7", 2, 0, 0)
    # run_chef("2_2", "8", 3, 1, 1)
    # run_chef("3", "9", 0, 0, 0)
    # run_chef("3_2", "10", 1, 1, 0)
    # run_chef("3", "11", 2, 0, 1)
    # run_chef("3", "12", 3, 1, 1)
    # run_chef("4", "13", 0, 0, 0)
    # run_chef("4", "14", 1, 0, 0)
    # run_chef("4", "15", 2, 0, 0)
    # run_chef("4_2", "16", 3, 1, 1)
