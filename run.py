import json
import numpy as np
import matplotlib.pyplot as plt

from problem.graph.double_coop_irl_from2 import Graph
# from problem.graph.data import GraphData
# from problem.graph.data2 import GraphData
# from problem.graph.data01 import GraphData
# from problem.graph.data03 import GraphData
from problem.graph.data04 import GraphData
# from problem.graph.train1 import GraphData
# from problem.graph.train2 import GraphData

import make_graph

def make_belief(size=2):
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    if size == 2:
        return np.concatenate(([b1], [b2]), axis=0).T
    b3 = np.zeros_like(b1)
    return np.concatenate(([b1], [b2], [b3]), axis=0).T



def run_chef():
    b = make_belief(2)

    graph = GraphData()
    env = Graph(graph)

    import time
    start = time.time()
    for d in [6]:
        env.calc_a_vector(d, b, 0)
        # env.calc_a_vector(0, d, b, with_a=True)

    scinario = env.make_scinario(0)
    # exit()
    json.dump(scinario, open("scinario.json", "w"), indent=4)
    make_graph.make_json(graph)

    # print(env.h_pi)
    # print(time.time() - start)
    # exit()
    for a_r in range(env.a_r):
        # v = np.array([env.value_a(0, a_r, b[i]) for i in range(len(b))])
        v = np.array([env.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        print(v)
    #     plt.plot(b[:, 0], v, label=a_r)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    run_chef()
