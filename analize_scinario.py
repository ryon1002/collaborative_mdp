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
    return np.mean(values), np.var(values)

def analize2(graph_id, ret, result_id, _algo, obj, pref):
    count = defaultdict(int)
    count_r = defaultdict(int)
    for v in ret.values():
        count["_".join(v[result_id]["H"])] += 1
        count_r["_".join(v[result_id]["R"])] += 1
    print(count)
    print(count_r)
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
        if user_id in ["a25b4155-6aec-5f1e-5871-40f84a0cd63c", "0988acc1-626d-aa25-e0b1-402710ae4fa5",
                       "4a966138-d31c-2357-849f-1141e473203b"]:
            continue
        # if user_id in ["4a966138-d31c-2357-849f-1141e473203b", "5cb4ae1d-125f-05f1-2649-1a88036a201c",
        #                "811defed-d9f2-9f31-8540-88b47246999c", "88c6459b-45c1-91dc-4372-bf1092dc09af",
        #                "0895e7ab-125f-1fbb-a616-2f66090a2382"]:
        # if user_id in ["4a966138-d31c-2357-849f-1141e473203b", "5cb4ae1d-125f-05f1-2649-1a88036a201c",
        #                "811defed-d9f2-9f31-8540-88b47246999c", "88c6459b-45c1-91dc-4372-bf1092dc09af",
        #                "0895e7ab-125f-1fbb-a616-2f66090a2382"]:
            continue
        data = json.loads(l[1])
        # if [int(d) for d in data[2:6]] == [1, 1, 1, 1]:
        #     continue
        # if [int(d) for d in data[2:6]] == [2, 2, 2, 2]:
        #     continue
        # if [int(d) for d in data[2:6]] == [3, 3, 3, 3]:
        #     continue
        # if [int(d) for d in data[2:6]] == [4, 4, 4, 4]:
        #     continue
        # if [int(d) for d in data[2:6]] == [5, 5, 5, 5]:
        #     continue
        # if [int(d) for d in data[2:6]] == [6, 6, 6, 6]:
        #     continue
        # if [int(d) for d in data[2:6]] == [7, 7, 7, 7]:
        #     continue
        # if int(data[2]) + 2 < int(data[5]):
        #     continue
        # if int(data[2]) + 2 >= int(data[5]):
        #     continue
        ret.append([int(d) for d in data[2:6]])
    ret = np.array(ret)
    ret2 = np.mean(ret, axis=0)

    ind_arr = np.repeat(list('ABCD'), len(ret))
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    print(pairwise_tukeyhsd(ret.T.flatten(), ind_arr, alpha=0.3))
    import scipy.stats

    # print(scipy.stats.f_oneway(ret[:,0], ret[:,1], ret[:,2], ret[:,3]))
    print(scipy.stats.ttest_rel(ret[:, 0], ret[:, 1]))
    print(scipy.stats.ttest_rel(ret[:, 0], ret[:, 2]))
    print(scipy.stats.ttest_rel(ret[:, 0], ret[:, 3]))
    print(scipy.stats.ttest_rel(ret[:, 1], ret[:, 2]))
    print(scipy.stats.ttest_rel(ret[:, 1], ret[:, 3]))
    print(scipy.stats.ttest_rel(ret[:, 2], ret[:, 3]))
    # print(scipy.stats.ttest_ind(ret[:, 0], ret[:, 1], equal_var=False))
    # print(scipy.stats.ttest_ind(ret[:, 0], ret[:, 2], equal_var=False))
    # print(scipy.stats.ttest_ind(ret[:, 0], ret[:, 3], equal_var=False))
    # print(scipy.stats.ttest_ind(ret[:, 1], ret[:, 2], equal_var=False))
    # print(scipy.stats.ttest_ind(ret[:, 1], ret[:, 3], equal_var=False))
    # print(scipy.stats.ttest_ind(ret[:, 2], ret[:, 3], equal_var=False))
    # print(scipy.stats.mannwhitneyu(ret[:, 0], ret[:, 1]))
    # print(scipy.stats.mannwhitneyu(ret[:, 0], ret[:, 2]))
    # print(scipy.stats.mannwhitneyu(ret[:, 0], ret[:, 3]))
    # print(scipy.stats.mannwhitneyu(ret[:, 1], ret[:, 2]))
    # print(scipy.stats.mannwhitneyu(ret[:, 1], ret[:, 3]))
    # print(scipy.stats.mannwhitneyu(ret[:, 2], ret[:, 3]))
    # print(ind_arr)
    # print(ret)

    # exit()

    color_map = {0:"red", 1:"green", 2:"blue", 3:"grey"}
    label_map = {0:"Ours(with Predictability Bias)", 1:"Ours(without Predictability Bias)",
                 2:"Optimistic CIRL", 3:"MinMax"}
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # plt.bar([0, 1, 2, 3], ret2, color=["red", "green", "blue", "grey"],
    #         label=["Ours(with Predictability Bias)", "Ours(without Predictability Bias)",
    #                "Optimistic CIRL", "MinMax"])
    # plt.bar([0, 1, 2, 3], ret2, color=["red", "green", "blue", "grey"],
    for i in range(4):
        plt.bar(i, ret2[i], color=color_map[i], label=label_map[i])
    plt.plot([2, 3], [4.5, 4.5], color="k")
    plt.plot([2, 2], [4.5, 4.3], color="k")
    plt.plot([3, 3], [4.5, 4.3], color="k")
    plt.plot([1, 3], [5, 5], color="k")
    plt.plot([3, 3], [5, 4.8], color="k")
    plt.plot([1, 1], [5, 4.8], color="k")
    plt.plot([0, 2], [5.5, 5.5], color="k")
    plt.plot([0, 0], [5.5, 5.3], color="k")
    plt.plot([2, 2], [5.5, 5.3], color="k")
    plt.plot([0, 3], [6, 6], color="k")
    plt.plot([0, 0], [6, 5.8], color="k")
    plt.plot([3, 3], [6, 5.8], color="k")
    plt.ylim([0, 9])
    plt.text(2.5, 4.5, "*")
    plt.text(2, 5, "*")
    plt.text(1, 5.5, "*")
    plt.text(1.5, 6, "**")
    plt.yticks(np.arange(8), np.arange(8))
    # plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1) )
    plt.legend(loc="upper right")
    plt.xticks([], [])
    plt.savefig("result_4.eps", bbox_inches="tight", format="eps")
    plt.show()
    exit()


    # exit()
    print(ret)
    print(ret.shape)


    print(np.mean(ret, axis=0))
    print(np.var(ret, axis=0))
    exit()

        # ret[user_id] = data
    return ret

def read_result3():
    import csv, json
    reader = csv.reader(open("result2.txt"), delimiter="\t")
    s = defaultdict(int)
    c_a = 0
    a = 0

    for l in reader:
        user_id = l[0]
        if user_id in ["a25b4155-6aec-5f1e-5871-40f84a0cd63c", "0988acc1-626d-aa25-e0b1-402710ae4fa5"]:
            continue
        data = json.loads(l[1])
        s[data[0]] += 1
        if data[1] != "X":
            c_a += 1
            a += int(data[1])
        # print(data)


    # print(s)
    # print(a / c_a)

if __name__ == '__main__':
    # ret = read_result()
    # print(ret)
    # exit()
    # ret = read_result2()
    # print(ret)
    # exit()
    # func = analize1
    ret = read_result2()
    exit()
    # print(ret)
    func = analize2
    # func("1", ret, "1", 0, 0, 0)
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
    # func("4_2", ret, "16", 3, 1, 1)
    exit()

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
    # func("4", ret, "13", 0, 0, 0)
    # func("4_2", ret, "14", 1, 0, 1)
    # func("4", ret, "15", 2, 1, 0)
    # func("4_2", ret, "16", 3, 1, 1)
