import numpy as np
import json
from collections import defaultdict

def calc_h_value(v):
    v = np.array(v)
    v[v < 0] = 500
    return np.min(v)

def make_worst(index, env):
    conv_action = {0: 2, 1: 1, 2: 4, 3: 3, 4: 0}
    s_queue = set(range(env.s - 1))

    counter = np.max([np.sum(env.ct_data.h_chip), np.sum(env.ct_data.r_chip)])
    h_values = np.zeros(env.s - 1)
    r_values = np.zeros(env.s - 1)

    tmp_actions = {}

    for c in range(counter, -1, -1):
        candidate = [s for s in s_queue if env.s_map[s][-1] == c]
        a_map = defaultdict(lambda: defaultdict(list))
        for n_s in candidate:
            for a_r, a_h, s in zip(*np.where(env.t[:, :, :, n_s] == 1)):
                # for th_
                # print(env.r[a_r, a_h, s, :, :])
                # exit()

                # a_map[s][a_r].append(env.r[a_r, a_h, s, :, :] + h_values[n_s])
                # a_map[s][a_r].append(np.mean(env.r[a_r, a_h, s, :, :], axis=0) + h_values[n_s])
                # print(env.s_map[n_s])
                # print(env.r[a_r, a_h, s, :, :])
                # print(np.mean(env.r[a_r, a_h, s, :, :], axis=0))
                a_map[s][a_r].append(np.min(env.r[a_r, a_h, s, :, :], axis=0) + h_values[n_s])
                # a_map[s][a_r].append(np.mean(env.r[a_r, a_h, s, :, :], axis=1) + h_values[n_s])


                # exit()
                # a_map[s][a_r].append(np.mean(env.r[a_r, a_h, s, :, :]) + value[n_s])
        for s, a_rh in a_map.items():
            # print(s, a_rh)
            # exit()

            # tmp_values = {a_r:np.min(v) for a_r, v in a_rh.items()}
            # tmp_values = {a_r:np.max(v) for a_r, v in a_rh.items()}
            tmp_values = {a_r:np.mean(np.max(np.array([i for i in v]), axis=0)) for a_r, v in a_rh.items()}
            # tmp_values = {a_r:calc_h_value(v) for a_r, v in a_rh.items()}
            # print(s, tmp_values)
            a, v = sorted(tmp_values.items(), key=lambda x:x[-1], reverse=True)[0]
            # print(a, v)
            h_values[s] = v
            tmp_actions[s] = a

            # h_values[s] = sorted(tmp_values.items(), key=lambda x:x[-1])[0][1]

        #     print(values)
        #     print(s, a_rh)
        # exit()
        # print(c)
    # for s in s_map:
    #     print(env.s_map[s])

    nexts = {}
    actions = {}
    for s, a_r in tmp_actions.items():
        next = {}
        for a_h, v in env.t_map[s].items():
            next[conv_action[a_h]] = v[a_r]
        nexts[int(s)] = next
        actions[int(s)] = conv_action[a_r]
    # print(actions[4])
    # print(nexts[4])
    json.dump({"actions": actions, "nexts": nexts},
              open("ct_data/scinario_" + str(index) + "_3.json", "w"), indent=4)

    # print(s_map)
