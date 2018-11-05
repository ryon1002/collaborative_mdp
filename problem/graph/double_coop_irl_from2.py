import itertools
import numpy as np
from model.double_coop_irl_from2 import CoopIRL


class Graph(CoopIRL):
    def __init__(self, graph_data):
        self.graph_data = graph_data
        a_h_list = [len(n) for n in graph_data.h_node]
        a_r_list = [len(n) for n in graph_data.r_node]

        h_counter = {i:0 for i in range(len(a_h_list))}
        self._count_state(h_counter, None, graph_data.h_edge, 0, len(a_h_list) - 1)
        r_counter = {i:0 for i in range(len(a_r_list))}
        self._count_state(r_counter, None, graph_data.r_edge, 0, len(a_r_list) - 1)
        s = 0
        for i in range(len(a_h_list)):
            s += h_counter[i] * r_counter[i]
        s += 1

        super().__init__(s, max(a_r_list), max(a_h_list),
                         len(graph_data.cost_candidate), len(graph_data.items))

    def _count_state(self, counter, node, edges, layer, limit):
        counter[layer] += 1
        if layer == limit:
            return
        for g in edges[node].keys():
            self._count_state(counter, g, edges, layer + 1, limit)


    def _calc_num_conbination(self, list):
        return int(sum([np.product(list[:i]) for i in range(len(list) + 1)]))

    def _check_complete(self, item, obj):
        lack = item - obj
        lack_ids = lack < 0
        lack_sums = np.sum(lack[lack_ids]) * -1
        if lack_sums == 0:
            return ()
        if lack_sums > 2:
            return (0, 0, 0)
        if lack_sums == 1:
            return (np.argmax(lack_ids),)
        if np.sum(lack_ids) == 1:
            l_id = np.argmax(lack_ids)
            return (l_id, l_id)
        return tuple(np.where(lack_ids)[0])

    def _set_tro(self):
        prev_states = {(): 0}
        prev_states_num = 1
        last_actions = {0: (None, None)}
        current_state = {}

        for i in range(len(self.graph_data.h_node)):
            h_node = self.graph_data.h_node[i]
            r_node = self.graph_data.r_node[i]
            for items, s in prev_states.items():
                p_ac_h, p_ac_r = last_actions[s]
                edge_h, edge_r = self.graph_data.h_edge[p_ac_h], self.graph_data.r_edge[p_ac_r]
                for a_h, a_r in itertools.product(range(self.a_h), range(self.a_r)):
                    ac_h = h_node[a_h] if a_h < len(h_node) else None
                    ac_r = r_node[a_r] if a_r < len(r_node) else None
                    if ac_h is None or ac_r is None or \
                        ac_h not in edge_h or ac_r not in edge_r:
                        self.t[a_r, a_h, s, -1] = 1
                        self.r[a_r, a_h, s, :, :] = -2000
                        continue

                    next_items = tuple(sorted(items + (ac_h, ac_r)))
                    if i == len(self.graph_data.h_node) - 1:
                        next_s = -1
                    else:
                        if next_items not in current_state:
                            current_state[next_items] = len(current_state) + prev_states_num
                        next_s = current_state[next_items]
                    self.t[a_r, a_h, s, next_s] = 1
                    last_actions[next_s] = (ac_h, ac_r)
                    ec_idx_h, ec_idx_r = edge_h[ac_h], edge_r[ac_r]
                    cost = np.sum(self.graph_data.cost_candidate[:, [ec_idx_h, ec_idx_r]], 1)

                    for th_h in range(self.th_h):
                        self.r[a_r, a_h, s, :, th_h] = cost
                        if ac_h in self.graph_data.items[th_h]:
                            self.r[a_r, a_h, s, :, th_h] += 400
                        if ac_r in self.graph_data.items[th_h]:
                            self.r[a_r, a_h, s, :, th_h] += 400
                        # if self.r[a_r, a_h, s, 0, th_h] > 100:
                        #     print(items, th_h, a_h, a_r)

            prev_states = current_state
            prev_states_num += len(prev_states)
            # print(current_state)
            current_state = {}

        # self.r = np.zeros((self.a_r, self.a_h, self.s, self.th_r, self.th_h))
        self.t[:, :, -1, -1] = 1

    def _make_one_turn(self, i, s, th_r, belief):
        print(i, s, th_r, belief)
        # print([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
        best_a_r = np.argmax([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
        next_map = {}
        for a_h in range(len(self.graph_data.h_node[i])):
            n_s = np.argmax(self.t[best_a_r, a_h, s])
            # print(n_s)
            if n_s == self.s - 1:
                next_map[self.graph_data.h_node[i][a_h]] = None
            else:
                n_b = belief.copy() * self.h_pi[th_r][s][best_a_r][a_h]
                next_map[self.graph_data.h_node[i][a_h]] = \
                    self._make_one_turn(i + 1, n_s, th_r, n_b)
        return (self.graph_data.r_node[i][best_a_r], next_map)
        # return (int(best_a_r), next_map)

    def make_scinario(self, th_r):
        # turn = len(self.graph_data.h_node)
        belief = np.array([0.5, 0.5])
        policy_map = {None : {}}
        # current_pos = policy_map[None]
        # self._make_one_turn(policy_map[None], 0, th_r, belief)
        policy_map = {None : self._make_one_turn(0, 0, th_r, belief)}
        # print(policy_map)
        # for i in range(turn):
        #     best_a_r = np.argmax([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
        #     print(best_a_r)
        #     for
        #
        #     # a_r =
        #     break
        return {"human_start":policy_map[None]}


