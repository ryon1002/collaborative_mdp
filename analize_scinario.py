import numpy as np
from problem.graph import train1, train2, data01, data01_2, data02, data02_2, data03, data04, data04_2
from collections import defaultdict

def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T

def calc_reward(graph, result, obj, pref):
    reward = 0
    for t, edge in zip(["H", "R"], [graph.h_edge, graph.r_edge]):
        seq = [None] + result[t]
        for i in range(len(seq) - 1):
            road = edge[seq[i]][seq[i + 1]]
            value = graph.cost_candidate[pref, road]
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


def analize1(graph_id, ret, result_id, _algo, obj, pref):
    values = [calc_reward(get_greph(graph_id), v[result_id], obj, pref) for v in ret.values()]
    print(result_id, np.mean(values), np.var(values))

def analize2(graph_id, ret, result_id, _algo, obj, pref):
    count = defaultdict(int)
    for v in ret.values():
        count["_".join(v[result_id]["H"])] += 1
    print(count)
    # print(result_id, np.mean(values), np.var(values))

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

def read_result2():
    import csv, json
    reader = csv.reader(open("result2.txt"), delimiter="\t")
    ret = []
    # next(reader)
    for l in reader:
        user_id = l[0]
        data = json.loads(l[1])
        ret.append([int(d) for d in data[2:6]])
    ret = np.array(ret)
    print(np.sum(ret, axis=0))

        # ret[user_id] = data
    return ret

if __name__ == '__main__':
    ret = read_result()
    # print(ret)
    # ret = read_result2()
    # print(ret)
    func = analize1
    func = analize2

    # func("1", ret, "1", 0, 0, 0)
    # func("1_2", ret, "2", 1, 1, 1)
    # func("1", ret, "3", 2, 1, 0)
    # func("1_2", ret, "4", 3, 0, 1)
    # func("2", ret, "5", 0, 0, 0)
    # func("2_2", ret, "6", 1, 0, 1)
    # func("2", ret, "7", 2, 1, 0)
    # func("2_2", ret, "8", 3, 1, 1)
    # func("3", ret, "9", 0, 0, 0)
    # func("3", ret, "10", 1, 1, 0)
    # func("3", ret, "11", 2, 0, 1)
    # func("3", ret, "12", 3, 1, 1)
    func("4", ret, "13", 0, 0, 0)
    func("4_2", ret, "14", 1, 0, 1)
    func("4", ret, "15", 2, 1, 0)
    func("4_2", ret, "16", 3, 1, 1)
