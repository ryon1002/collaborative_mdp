import numpy as np
from problem.graph import train1, train2, data01, data01_2, data02, data02_2, data03, data04, data04_2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T

def calc_reward(graph, result, obj, pref, cost):
    reward = 0
    for t, edge in zip(["H", "R"], [graph.h_edge, graph.r_edge]):
    # for t, edge in zip(["R"], [graph.r_edge]):
    # for t, edge in zip(["H"], [graph.h_edge]):
        seq = [None] + result[t]
        for i in range(len(seq) - 1):
            road = edge[seq[i]][seq[i + 1]]
            value = cost[pref, road]
            reward += value
            if seq[i +1] in graph.items[obj]:
                reward += 400
    return reward

def get_greph(graph_id):
    if graph_id == "t1" : graph = train1.GraphData()
    elif graph_id == "t2" : graph = train2.GraphData()
    elif graph_id == "1" : graph = data01.GraphData()
    elif graph_id == "1_2" : graph = data01_2.GraphData()
    elif graph_id == "2" : graph = data02.GraphData()
    elif graph_id == "2_2" : graph = data02_2.GraphData()
    elif graph_id == "3" : graph = data03.GraphData()
    elif graph_id == "4" : graph = data04.GraphData()
    elif graph_id == "4_2" : graph = data04_2.GraphData()
    return graph


def analize1(graph_id, ret, result_id, algo, obj, pref, **kwargs):
    cost = np.array([[0, -100, -20, 400], [0, -20, -100, 400]])
    color_map = {0:"red", 1:"green", 2:"blue", 3:"grey"}
    label_map = {0:"Ours(with Predictability Bias)", 1:"Ours(without Predictability Bias)",
                 2:"Optimistic CIRL", 3:"MinMax"}
    values = [calc_reward(get_greph(graph_id), v[result_id], obj, pref, cost) for v in ret.values()]
    if "best_road" in kwargs:
        e_reward = 0
        for n, i in enumerate(kwargs["best_road"]):
            e_reward += cost[0][n] * i
        print(result_id, np.mean(values), np.std(values), e_reward, np.mean(values) / e_reward, np.std(values / e_reward))
    else:
        print(result_id, np.mean(values), np.std(values))
    # plt.bar(result_id, np.mean(values))

    # print(values)
    graph= int(graph_id[0])
    score = np.mean(values)
    err = np.std(values)
    # score = np.mean(np.mean(values) / e_reward)
    # err = np.std(values / e_reward)
    # print(score, err)
    # print(values / e_reward)
    # score = np.std(values)
    # score = np.mean(values) /e_reward
    if graph == 1:
        plt.bar(graph + (algo + 1) * 0.15, score, yerr=err, width=0.15, color=color_map[algo], label=label_map[algo])
        # plt.bar(graph + (algo + 1) * 0.15, score, width=0.15, color=color_map[algo], label=label_map[algo])
    else:
        graph -= 1
        plt.bar(graph + (algo + 1) * 0.15, score, yerr=err, width=0.15, color=color_map[algo])
        # plt.bar(graph + (algo + 1) * 0.15, score, width=0.15, color=color_map[algo])
    return np.mean(values), np.var(values)

def analize2(graph_id, ret, result_id, _algo, obj, pref):
    count = defaultdict(int)
    for v in ret.values():
        count["_".join(v[result_id]["H"])] += 1
    # print(count)
    print(result_id, np.mean(values), np.var(values))

def check_valid_user(data):
    # h_g = [v["H"][-1][-1] for v in data.values()]
    # print(np.sum([i == "a" for i in h_g]))
    return True

def read_result():
    import csv, json
    reader = csv.reader(open("result.txt"), delimiter="\t")
    ret = {}
    # next(reader)
    for l in reader:
        user_id = l[0]
        data = json.loads(l[1])
        if check_valid_user(data):
            ret[user_id] = data
    return ret

def plot_oracle(id, road):
    cost = np.array([[0, -100, -20, 400], [0, -20, -100, 400]])
    e_reward = 0
    for n, i in enumerate(road):
        e_reward += cost[0][n] * i
    if id == 1:
        plt.bar(id, e_reward, width=0.15, color="yellow", label="Oracle")
    else :
        plt.bar(id, e_reward, width=0.15, color="yellow")

if __name__ == '__main__':
    ret = read_result()
    func = analize1


    plot_oracle(1, [0, 0, 2, 3])
    plot_oracle(2, [0, 0, 4, 3])
    plot_oracle(3, [0, 0, 4, 3])
    func("1", ret, "1", 0, 0, 0, best_road=[0, 0, 4, 3])
    func("1_2", ret, "2", 1, 1, 1, best_road=[0, 1, 3, 3])
    func("1", ret, "3", 2, 1, 0, best_road=[0, 0, 3, 3])
    func("1_2", ret, "4", 3, 0, 1, best_road=[0, 1, 1, 2])
    func("3", ret, "9", 0, 0, 0, best_road=[0, 0, 5, 3])
    func("3", ret, "10", 1, 1, 0, best_road=[0, 0, 5, 3])
    func("3", ret, "11", 2, 0, 1, best_road=[0, 0, 4, 3])
    func("3", ret, "12", 3, 1, 1, best_road=[0, 0, 4, 3])
    func("4", ret, "13", 0, 0, 0, best_road=[0, 3, 0, 3])
    func("4_2", ret, "14", 1, 0, 1, best_road=[0, 0, 4, 3])
    func("4", ret, "15", 2, 1, 0, best_road=[0, 0, 4, 3])
    func("4_2", ret, "16", 3, 1, 1, best_road=[0, 0, 4, 2.5])
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.ylabel("Sd of Average Reward")
    # plt.ylim([0, 650])
    plt.xlabel("Scinario")

    plt.ylabel("Avarage Reward")
    plt.xticks([1.3, 2.3, 3.3], ["(1)", "(2)", "(3)"])
    plt.ylim([0, 2000])
    # plt.ylim([0, 1700])

    # plt.ylabel("Average reward / Estimated Reward")
    # plt.xticks([1.38, 2.38, 3.38], ["(1)", "(2)", "(3)"])
    # # plt.ylim([0, 1.75])
    # plt.ylim([0, 1.4])
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.legend(loc="upper right")
    # plt.savefig("result_1.eps", bbox_inches="tight", format="eps")
    plt.savefig("result_1_err.eps", bbox_inches="tight", format="eps")
    # plt.savefig("result_3.eps", bbox_inches="tight", format="eps")
    # plt.savefig("result_3_err.eps", bbox_inches="tight", format="eps")
    plt.show()


