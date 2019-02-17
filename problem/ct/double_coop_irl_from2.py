import itertools
import numpy as np
# from model.double_coop_irl_from2 import CoopIRL
from model.double_coop_irl_from2_bak3 import CoopIRL
from collections import defaultdict
import json


class ColorTrails(CoopIRL):
    action = {0: np.array([-1, 0]),  # up
              1: np.array([0, -1]),  # left
              2: np.array([0, 1]),  # right
              3: np.array([1, 0])}  # down

    def __init__(self, ct_data):
        self.ct_data = ct_data
        self.t_map = defaultdict(lambda: defaultdict(dict))
        self.s_map = {}

        self.make_s_map_for_r(0, set(), self.s_map, self.t_map, ct_data.h_start, ct_data.r_start,
                              ct_data.h_chip, ct_data.r_chip, 0)
        super().__init__(len(self.s_map) + 1, 5, 5, 2, 2)

    def valid_pos(self, pos):
        c = (pos >= np.array([0, 0])) * (pos < np.array(self.ct_data.shape))
        return np.prod(c)

    def make_s_map_for_r(self, s, medals, s_map, t_map, h_pos, r_pos, h_chip, r_chip, d):
        medals = medals.copy()
        if tuple(h_pos) in self.ct_data.medals:
            medals.add(self.ct_data.medals[tuple(h_pos)])
        if tuple(r_pos) in self.ct_data.medals:
            medals.add(self.ct_data.medals[tuple(r_pos)])

        bomb = -1
        if tuple(h_pos) in self.ct_data.bomb:
            bomb = self.ct_data.bomb[tuple(h_pos)]

        s_map[s] = (h_pos, r_pos, h_chip, r_chip, medals, self._finish_recipe(medals), bomb, d)
        s_offset = 0
        r_halt = True
        if tuple(r_pos) not in self.ct_data.medals:
            for a_r in range(4):
                n_r_pos = r_pos + ColorTrails.action[a_r]
                if not self.valid_pos(n_r_pos):
                    continue
                chip = self.ct_data.color[n_r_pos[0], n_r_pos[1]]
                if r_chip[chip] <= 0:
                    continue
                r_halt = False
                n_r_chip = r_chip.copy()
                n_r_chip[chip] -= 1
                s_offset += self.make_s_map_for_h(s, medals, a_r, s_offset, s_map, t_map,
                                                  h_pos, n_r_pos, h_chip, n_r_chip, d)
        if r_halt:
            s_offset += self.make_s_map_for_h(s, medals, 4, s_offset, s_map, t_map,
                                              h_pos, r_pos, h_chip, r_chip, d)

        return s_offset

    def make_s_map_for_h(self, s, medals, a_r, s_offset, s_map, t_map,
                         h_pos, n_r_pos, h_chip, n_r_chip, d):
        h_halt = True
        s_h_offset = 0
        if tuple(h_pos) not in self.ct_data.medals:
            for a_h in range(4):
                n_h_pos = h_pos + ColorTrails.action[a_h]
                if not self.valid_pos(n_h_pos):
                    continue
                chip = self.ct_data.color[n_h_pos[0], n_h_pos[1]]
                if h_chip[chip] <= 0:
                    continue
                h_halt = False
                n_h_chip = h_chip.copy()
                n_h_chip[chip] -= 1
                s_h_offset += 1

                # t_map[s].append((a_h, a_r, s + s_offset + s_h_offset))
                t_map[s][a_h][a_r] = s + s_offset + s_h_offset
                s_h_offset += self.make_s_map_for_r(s + s_offset + s_h_offset, medals, s_map, t_map,
                                                    n_h_pos, n_r_pos, n_h_chip, n_r_chip, d + 1)
        if h_halt and a_r != 4:
            s_h_offset += 1
            t_map[s][4][a_r] = s + s_offset + s_h_offset
            s_h_offset += self.make_s_map_for_r(s + s_offset + s_h_offset, medals, s_map, t_map,
                                                h_pos, n_r_pos, h_chip, n_r_chip, d + 1)
        return s_h_offset

    def _finish_recipe(self, medals):
        recipe = set()
        for i, rs in enumerate(self.ct_data.recipe):
            for r in rs:
                if r.issubset(medals):
                    recipe.add(i)
                    break
        return recipe

    def _set_tro(self):
        for s, v in self.t_map.items():
            for a_h in range(5):
                if a_h not in v:
                    self.t[:, a_h, s, -1] = 1
                    self.r[:, a_h, s, :, :] = -1000
                else:
                    for a_r in range(5):
                        if a_r not in v[a_h]:
                            self.t[a_r, a_h, s, -1] = 1
                            self.r[a_r, a_h, s, :, :] = -1000
                        else:
                            ns = v[a_h][a_r]
                            self.t[a_r, a_h, s, ns] = 1
                            self.r[a_r, a_h, s, :, :] -= self.calc_cost(a_h, a_r)
                            s_data = self.s_map[s]
                            ns_data = self.s_map[ns]
                            for recipe in ns_data[5]:
                                if recipe not in s_data[5]:
                                    self.r[a_r, a_h, s, :, recipe] += 300
                            if ns_data[6] != -1:
                                self.r[a_r, a_h, s, ns_data[6], :] -= 100
        for s in self.s_map.keys():
            if s not in self.t_map:
                self.t[:, :, s, -1] = 1
                self.r[:-1, :, s, :] = -1000
                self.r[:, :-1, s, :] = -1000
        self.t[:, :, -1, -1] = 1

        # for k, v in self.s_map.items():
        #     print(k, v)
        # # exit()
        #
        # for k, v in self.t_map.items():
        #     print(k, v)
        # exit()

    def calc_cost(self, a_h, a_r):
        return (int(a_h != 4) + int(a_r != 4)) * 5

    def make_data(self, index):
        # medals = {v: k for k, v in self.ct_data.medals.items()}
        medals = np.zeros_like(self.ct_data.color)
        for k, v in self.ct_data.medals.items():
            medals[k] = v + 1
        bomb = np.zeros_like(self.ct_data.color)
        for k, v in self.ct_data.bomb.items():
            bomb[k] = v + 1
        recipe = [[list(r) for r in rs] for rs in self.ct_data.recipe]
        data = {
            "color": self.ct_data.color.tolist(),
            "bomb": bomb.tolist(),
            "medals": medals.tolist(),
            "h_chip": self.ct_data.h_chip.tolist(),
            "r_chip": self.ct_data.r_chip.tolist(),
            "recipe": recipe,
            "h_start": self.ct_data.h_start.tolist(),
            "r_start": self.ct_data.r_start.tolist(),
        }
        json.dump(data, open("ct_data/data_" + str(index) + ".json", "w"), indent=4)

    def make_scinario(self, th_r, index, algo):
        conv_action = {0: 2, 1: 1, 2: 4, 3: 3, 4: 0}
        s_candi = set([0])
        b_map = {0: np.array([0.5, 0.5])}
        actions = {}
        nexts = {}
        while len(s_candi) > 0:
            s = s_candi.pop()
            b = b_map[s]
            a_r = self.a_vector_a[s][th_r]
            # print(a_r)
            # return np.max(np.dot(self.a_vector_a[s][th_r][a_r], b))
            # print(s, [np.dot(b, v.T)[0][0] for _k, v in sorted(a_r.items())])
            # print(s, [v for _k, v in sorted(a_r.items())])
            # print(s, [np.max(np.dot(b, v.T)) for _k, v in sorted(a_r.items())])
            # exit()
            # print(s, [np.dot(b, v.T) for _k, v in sorted(a_r.items())])
            # print(s, np.max(np.dot(b, v.T)[0])[0] for _k, v in sorted(a_r.items())])
            a_r = np.argmax([np.max(np.dot(b, v.T)) for _k, v in sorted(a_r.items())])

            # print(s, a_r)
            next = {}
            for a_h, v in self.t_map[s].items():
                n_s = v[a_r]
                b = self.h_pi[th_r][s][a_r][a_h] * b_map[s]
                b /= np.sum(b)
                b_map[n_s] = np.array(b)
                next[conv_action[a_h]] = n_s
                s_candi.add(n_s)
            if len(next) > 0:
                nexts[s] = next
            actions[s] = int(conv_action[a_r])
        # print(actions)
        json.dump({"actions": actions, "nexts": nexts},
                  open("ct_data/scinario_" + str(index) + "_" + str(algo) + ".json", "w"), indent=4)
        # json.dump(actions, open("ct_data/scinario_" + str(index) + "_" + str(algo) + ".json", "w"), indent=4)

        # print(b)
        # print(self.h_pi[th_r][s][a_r])

    def _take_one_turn(self):
        exit()

    # def _count_state(self, counter, node, edges, layer, limit):
    #     counter[layer] += 1
    #     if layer == limit:
    #         return
    #     for g in edges[node].keys():
    #         self.k_count_state(counter, g, edges, layer + 1, limit)
    #
    # def _calc_num_conbination(self, list):
    #     return int(sum([np.product(list[:i]) for i in range(len(list) + 1)]))
    #
    # def _check_complete(self, item, obj):
    #     lack = item - obj
    #     lack_ids = lack < 0
    #     lack_sums = np.sum(lack[lack_ids]) * -1
    #     if lack_sums == 0:
    #         return ()
    #     if lack_sums > 2:
    #         return (0, 0, 0)
    #     if lack_sums == 1:
    #         return (np.argmax(lack_ids),)
    #     if np.sum(lack_ids) == 1:
    #         l_id = np.argmax(lack_ids)
    #         return (l_id, l_id)
    #     return tuple(np.where(lack_ids)[0])
    #
    # def _set_tro(self):
    #     prev_states = {(): 0}
    #     prev_states_num = 1
    #     last_actions = {0: (None, None)}
    #     current_state = {}
    #
    #     for i in range(len(self.graph_data.h_node)):
    #         for items, s in prev_states.items():
    #             p_ac_h, p_ac_r = last_actions[s]
    #             edge_h, edge_r = self.graph_data.h_edge[p_ac_h], self.graph_data.r_edge[p_ac_r]
    #             h_node = sorted(edge_h.keys())
    #             r_node = sorted(edge_r.keys())
    #
    #             for a_h, a_r in itertools.product(range(self.a_h), range(self.a_r)):
    #                 ac_h = h_node[a_h] if a_h < len(h_node) else None
    #                 ac_r = r_node[a_r] if a_r < len(r_node) else None
    #                 if ac_h is None or ac_r is None or \
    #                         ac_h not in edge_h or ac_r not in edge_r:
    #                     self.t[a_r, a_h, s, -1] = 1
    #                     self.r[a_r, a_h, s, :, :] = -2000
    #                     continue
    #
    #                 next_items = tuple(sorted(items + (ac_h, ac_r)))
    #                 if i == len(self.graph_data.h_node) - 1:
    #                     next_s = -1
    #                 else:
    #                     if next_items not in current_state:
    #                         current_state[next_items] = len(current_state) + prev_states_num
    #                     next_s = current_state[next_items]
    #                 self.t[a_r, a_h, s, next_s] = 1
    #                 last_actions[next_s] = (ac_h, ac_r)
    #                 ec_idx_h, ec_idx_r = edge_h[ac_h], edge_r[ac_r]
    #                 cost = np.sum(self.graph_data.cost_candidate[:, [ec_idx_h, ec_idx_r]], 1)
    #
    #                 for th_h in range(self.th_h):
    #                     self.r[a_r, a_h, s, :, th_h] = cost
    #                     if ac_h in self.graph_data.items[th_h]:
    #                         self.r[a_r, a_h, s, :, th_h] += 400
    #                     if ac_r in self.graph_data.items[th_h]:
    #                         self.r[a_r, a_h, s, :, th_h] += 400
    #
    #         prev_states = current_state
    #         prev_states_num += len(prev_states)
    #         self.state_map.update(current_state)
    #         current_state = {}
    #
    #     self.t[:, :, -1, -1] = 1

    # def _make_one_turn(self, i, s, th_r, belief, last_r_node, last_h_node):
    #     values = np.array([self.value_a(s, th_r, a_r, belief) for a_r in range(self.a_r)])
    #     max_values = np.max(values)
    #
    #     best_a_rs = np.where(values == max_values)[0]
    #
    #     if len(best_a_rs) > 1:
    #         values_1 = np.array([self.value_a(s, th_r, a_r, [1, 0]) for a_r in range(self.a_r)])
    #         values_2 = np.array([self.value_a(s, th_r, a_r, [0, 1]) for a_r in range(self.a_r)])
    #         best_a_rs = [np.argmax(values + values_1 + values_2)]
    #     ret = []
    #
    #     for best_a_r in best_a_rs:
    #         next_map = {}
    #         h_node = sorted(self.graph_data.h_edge[last_h_node].keys())
    #         r_node = sorted(self.graph_data.r_edge[last_r_node].keys())
    #         for a_h in range(len(h_node)):
    #             n_s = np.argmax(self.t[best_a_r, a_h, s])
    #             if n_s == self.s - 1:
    #                 next_map[h_node[a_h]] = None
    #             else:
    #                 n_b = belief.copy() * self.h_pi[th_r][s][best_a_r][a_h]
    #                 next_map[h_node[a_h]] = self._make_one_turn(i + 1, n_s, th_r, n_b,
    #                                                             r_node[best_a_r], h_node[a_h])
    #         ret.append((r_node[best_a_r], next_map))
    #     return ret
    #
    # def make_scinario(self, th_r):
    #     belief = np.array([0.5, 0.5])
    #     policy_map = {None: {}}
    #     # current_pos = policy_map[None]
    #     # self._make_one_turn(policy_map[None], 0, th_r, belief)
    #     policy_map = {None: self._make_one_turn(0, 0, th_r, belief, None, None)}
    #     return {"human_start": policy_map[None]}
