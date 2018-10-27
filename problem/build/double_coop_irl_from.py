import itertools
import numpy as np
from model.double_coop_irl_from import CoopIRL


class Build(CoopIRL):
    def __init__(self, build_data):
        self.build_data = build_data
        a_h_list = [len(n) for n in build_data.h_node]
        a_r_list = [len(n) for n in build_data.r_node]

        s = 0
        for i in range(len(a_h_list)):
            s += np.prod(a_h_list[:i]) * np.prod(a_r_list[:i])
        s += 1
        s = int(s)

        super().__init__(s, max(a_r_list), max(a_h_list),
                         len(build_data.cost_candidate), len(build_data.recipe))

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
        current_state = {}

        for i in range(len(self.build_data.h_node)):
            h_node = self.build_data.h_node[i]
            r_node = self.build_data.r_node[i]
            for items, s in prev_states.items():
                for a_h, a_r in itertools.product(range(self.a_h), range(self.a_r)):
                    ac_h = h_node[a_h] if a_h < len(h_node) else None
                    ac_r = r_node[a_r] if a_r < len(r_node) else None
                    if ac_h is None or ac_r is None:
                        self.t[a_r, a_h, s, -1] = 1
                        self.r[a_r, a_h, s, :, :] = -1000
                        continue

                    next_items = tuple(sorted(items + (ac_h, ac_r)))
                    if i == len(self.build_data.h_node) - 1:
                        next_s = -1
                    else:
                        if next_items not in current_state:
                            current_state[next_items] = len(current_state) + prev_states_num
                        next_s = current_state[next_items]
                    self.t[a_r, a_h, s, next_s] = 1
                    action_idxs = [self.build_data.action_index[a] for a in (ac_h, ac_r)]
                    cost = np.sum(self.build_data.cost_candidate[:, action_idxs], 1)
                    for th_h in range(self.th_h):
                        self.r[a_r, a_h, s, :, th_h] = cost
                        if next_items in self.build_data.recipe_set[th_h]:
                            self.r[a_r, a_h, s, :, th_h] += 100
                            # print(next_items, th_h)
                            # print(s, a_r, a_h, self.r[a_r, a_h, s, :, th_h])
            prev_states = current_state
            prev_states_num += len(prev_states)
            # print(current_state)
            current_state = {}

        self.t[:, :, -1, -1] = 1

        # exit()


