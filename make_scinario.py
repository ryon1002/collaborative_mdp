import json
import os
import numpy as np
import matplotlib.pyplot as plt

from problem.graph.double_coop_irl_from2 import Graph
from problem.graph import train1, train2, data01

import make_graph

def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T


def run_chef(graph_id, dir_name, algo, obj, pref):
    if graph_id == "t1" : graph = train1.GraphData()
    elif graph_id == "t2" : graph = train2.GraphData()
    elif graph_id == "1" : graph = data01.GraphData()

    dir_name = "scinario/" + dir_name + "/"
    os.makedirs(dir_name, exist_ok=True)
    b = make_belief()
    env = Graph(graph)

    for d in [6]:
        env.calc_a_vector(d, b, algo)

    scinario = env.make_scinario(pref)
    json.dump(scinario, open(dir_name + "scinario.json", "w"), indent=4)
    json_data = make_graph.make_json(graph, algo, obj)
    json.dump(json_data, open(dir_name + "data.json", "w"), indent=4)

    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        # print(v)
    #     plt.plot(b[:, 0], v, label=a_r)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    run_chef("t1", "train1", 0, 0, 0)
    run_chef("t2", "train2", 0, 0, 0)
    run_chef("1", "1", 0, 0, 0)
    run_chef("1", "2", 1, 0, 0)
    run_chef("1", "3", 2, 0, 0)
