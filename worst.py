import numpy as np


def _make_one_turn(th_r, graph, last_r_node, last_h_node, r_cost):
    # print(i, s, th_r, belief, last_r_node, last_h_node)
    # print([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
    # values = np.array([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
    # max_values = np.max(values)
    # print(last_r_node, last_h_node)
    item0 = int(last_r_node in graph.items[0]) + int(last_r_node in graph.items[0])
    item1 = int(last_r_node in graph.items[1]) + int(last_r_node in graph.items[1])
    value = min([item0, item1]) * 200

    if last_r_node not in graph.r_edge:
        return (last_r_node, None), value + r_cost
    best_cost = -1000
    ret = None
    for a_r, r_r in graph.r_edge[last_r_node].items():
        next_map = {}
        next_cost = []
        for a_h, r_h in graph.h_edge[last_h_node].items():
            cost = np.sum((graph.cost_candidate[th_r, [r_r, r_h]]))
            m, c = _make_one_turn(th_r, graph, a_r, a_h, cost)
            next_map[a_h] = m
            next_cost.append(c)
        next_cost = min(next_cost)
        if next_cost > best_cost:
            ret = (a_r, next_map), next_cost
        # print(last_h_node, last_r_node, a_r, next_map, cost)
    return ret


    # if sum(values == max_values) == 1:
    #     best_a_r = np.argmax([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
    # else :
    #     values_1 = np.array([self.value_a(s, th_r, a_r, [1, 0]) for a_r in range(self.a_r)])
    #     values_2 = np.array([self.value_a(s, th_r, a_r, [0, 1]) for a_r in range(self.a_r)])
    #     best_a_r = np.argmax(values + values_1 + values_2)
    # next_map = {}
    # h_node = sorted(self.graph_data.h_edge[last_h_node].keys())
    # r_node = sorted(self.graph_data.r_edge[last_r_node].keys())
    # for a_h in range(len(h_node)):
    #     n_s = np.argmax(self.t[best_a_r, a_h, s])
    #     # print(n_s)
    #     if n_s == self.s - 1:
    #         # print(s)
    #         # exit()
    #         next_map[h_node[a_h]] = None
    #     else:
    #         n_b = belief.copy() * self.h_pi[th_r][s][best_a_r][a_h]
    #         next_map[h_node[a_h]] = self._make_one_turn(i + 1, n_s, th_r, n_b, r_node[best_a_r], h_node[a_h])
    # return (r_node[best_a_r], next_map)
    # return (int(best_a_r), next_map)

def make_worst(th_r, graph):
    policy_map = {None : {}}
    policy_map = {None : _make_one_turn(th_r, graph, None, None, 0)[0]}
    return {"human_start":policy_map[None]}


