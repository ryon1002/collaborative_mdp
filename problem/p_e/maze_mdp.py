from algo.vi import do_value_iteration
from model.coop_irl_mdp import CoopIRLMDP
import numpy as np
import copy
import itertools
import json

a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}

class MazeMDP(CoopIRLMDP):
    def __init__(self, maze, d=0):
        self.s_count = 0
        self.s_map = {}
        self.maze = maze
        h_actions = maze.nodes[maze.state.human].nexts.keys()
        init_state = copy.deepcopy(maze.state)
        for a_h in h_actions:
            maze.state = copy.deepcopy(init_state)
            maze.move_h(a_h)
            history = ((maze.state.prev_human, maze.state.human),(maze.state.agent,))
            self.search_state(maze, 0, d, (), [(a_h, -1)], history)
        th_h = 2 if maze.r_enemy_num > 0 else 1
        super().__init__(len(self.s_map) + 1, 4, 4, maze.b_enemy_num, th_h)
        maze.state = init_state

    def get_a_list(self, maze, history):
        a_list = []
        for a in maze.possible_action():
            n_h = tuple(np.array(maze.state.human) + a_dir[a[0]])
            if n_h in history[0]:
                continue
            n_a = tuple(np.array(maze.state.agent) + a_dir[a[1]])
            if n_a in history[1]:
                continue
            a_list.append(a)
        return a_list


    def search_state(self, maze, s, d, last_a, last_actions, history):
        if d == 0 or maze.state.done != -1:
            if maze.state.done != -1:
                print(maze.state.done, d)
                print(last_actions)
                # maze.show_world()
            return None, s, maze.state.done, last_a
        # a_list = [a for a in maze.possible_action()
        #           if not self.is_inv_action(last_actions[-1][0], a[0]) and
        #           not self.is_inv_action(last_actions[-1][1], a[1])]
        a_list = self.get_a_list(maze, history)
        # print(list(maze.possible_action()), maze.state.action)
        if len(a_list) == 0:
            return None, s, -1, last_a
        elif len(a_list) == 1:
            a = a_list[0]
            maze.move_ah(*a)
            all_a = last_a + (a,)
            next_history = (history[0] + (maze.state.human,), history[1] + (maze.state.agent, ))
            # print(history)
            # print(next_history)
            # exit()
            end, end_s, done, all_a = self.search_state(maze, s, d - 1, all_a, last_actions + [a],
                                                        next_history)
            return end, end_s, done, all_a
        else:
            state = copy.deepcopy(maze.state)
            self.s_map[s] = {}
            end_s = s + 1
            for a in a_list:
                maze.state = copy.deepcopy(state)
                maze.move_ah(*a)
                next_history = (history[0] + (maze.state.human,), history[1] + (maze.state.agent, ))
                # print(history)
                # print(next_history)
                # exit()
                end, end_s, done, all_a = self.search_state(maze, end_s, d - 1, (a,),
                                                            last_actions + [a], next_history)
                self.s_map[s][all_a] = (end, done)
        return s, end_s, -1, last_a

    def is_inv_action(self, a1, a2):
        return a1 + a2 == 3

    def _set_tro(self):
        self.t[:, :, -1, -1] = 1
        # self.r[:, :, :, :, :] = 0
        self.r[:, :, -1, :, :] = 0
        for s, nexts in self.s_map.items():
            nexts_f = {k[0]: (v, len(k)) for k, v in nexts.items()}
            for a_h, a_r in itertools.product(range(4), range(4)):
                if (a_h, a_r) in nexts_f:
                    (n_s, done), l = nexts_f[(a_h, a_r)]
                    if n_s is not None:
                        self.t[a_r, a_h, s, n_s] = 1
                        self.r[a_r, a_h, s, :, :] = -l
                        # self.r = np.zeros((self.a_r, self.a_h, self.s, self.th_r, self.th_h))
                    else:
                        self.t[a_r, a_h, s, -1] = 1
                        self.r[a_r, a_h, s, :, :] = -l
                        if done != -1:
                            self.r[a_r, a_h, s, done % 10, done // 10] += 100
                else:
                    self.t[a_r, a_h, s, -1] = 1
                    self.r[a_r, a_h, s, :, :] = -1000

    def make_single_policy(self):
        self.single_t = np.zeros((self.a_r * self.a_h, self.s, self.s))
        self.single_r = np.zeros((self.th_r, self.th_h, self.a_r * self.a_h, self.s, self.s))
        for a_r in range(self.a_r):
            for a_h in range(self.a_h):
                self.single_t[self.a_h * a_r + a_h] = self.t[a_r, a_h]
                for th_r in range(self.th_r):
                    for th_h in range(self.th_h):
                        for s in range(self.s):
                            self.single_r[th_r, th_h, self.a_h * a_r + a_h, s, :] \
                                = self.r[a_r, a_h, s, th_r, th_h]

        self.single_q = np.zeros((self.th_r, self.th_h, self.a_r, self.s))
        for th_r in range(self.th_r):
            for th_h in range(self.th_h):
                q = do_value_iteration(self.single_t, self.single_r[th_r, th_h])
                for a_r in range(self.a_r):
                    self.single_q[th_r, th_h, a_r] = np.max(q[a_r * self.a_h:(a_r + 1) * self.a_h],
                                                            axis=0)

    def make_scinario(self, file_name, irl):
        data = {}
        map = self.maze.map.copy()
        map[map > 1] = 1
        data["map"] = map.tolist()

        # self.pos_s_count = 0
        data["pos"] = {0: self._s_data(self.maze.state)}
        data["t"] = {}
        # a_h = list(self.maze.nodes[self.maze.state.human].nexts.keys())[0]
        self.search_state_for_scinario(0, 0, [
            list(self.maze.nodes[self.maze.state.human].nexts.keys())[0]], -1, data["pos"],
                                       data["t"], irl)
        # self.maze.move_h(a_h)
        # for a_h in h_actions:

        json.dump(data, open(file_name, "w"), indent=2)

    def _s_data(self, state):
        return (state.human, state.agent, tuple(state.b_enemys), tuple(state.r_enemys))

    def search_state_for_scinario(self, s, pos_s, a_h_list, a_r, pos, t, irl, d=3):
        # if d == 0:
        #     return
        # irl.
        # if s = 0:
        b = np.array([1.0])
        state = copy.deepcopy(self.maze.state)
        t[pos_s] = {}
        for a_h in a_h_list:
            prev_pos_s = pos_s
            self.maze.state = copy.deepcopy(state)
            ns = np.argmax(self.t[a_r, a_h, s]) if a_r != -1 else s
            v = np.array([[irl.value_a(ns, th_r, a_r, b) for a_r in range(self.a_r)]
                          for th_r in range(self.th_r)])
            n_a_r = np.argmax(np.max(v, axis=0))
            if ns == 0:
                in_a_list = [(a_h, -1), (None, n_a_r)]
            else:
                for in_a_list in self.s_map[s].keys():
                    if in_a_list[0] == (a_h, a_r):
                        break
                else:
                    print("error")
                    exit()
                in_a_list += ((None, n_a_r), )
            for in_a_i in range(len(in_a_list) - 1):
                in_a = in_a_list[in_a_i][0], in_a_list[in_a_i + 1][1]
                if in_a_i == len(in_a_list) - 2 and ns == self.s - 1:
                    self.maze.move_only_a(in_a[0])
                else:
                    self.maze.move_ah(in_a[0], in_a[1])
                pos[len(pos)] = self._s_data(self.maze.state)
                if prev_pos_s not in t:
                    t[prev_pos_s] = {}
                t[prev_pos_s][in_a[0]] = len(pos) - 1
                prev_pos_s = len(pos) - 1
            if ns != self.s - 1:
                a_h_list_2 = set()
                for a in self.s_map[ns].keys():
                    if a[0][1] == n_a_r:
                        a_h_list_2.add(a[0][0])
                self.search_state_for_scinario(ns, len(pos) - 1, list(a_h_list_2),
                                               n_a_r, pos, t, irl, d - 1)
