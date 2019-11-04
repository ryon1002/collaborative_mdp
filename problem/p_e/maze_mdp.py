from algo.vi import do_value_iteration
from model.coop_irl_mdp import CoopIRLMDP
import numpy as np
import copy
import itertools
import json


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
            self.search_state(maze, self.s_count, d, [(a_h, -1)])
            if len(self.s_map[self.s_count]) == 0:
                self.s_map.pop(self.s_count)
                self.s_count -= 1
            self.s_count += 1
        th_h = 2 if maze.r_enemy_num > 0 else 1
        super().__init__(len(self.s_map) + 1, 4, 4, maze.b_enemy_num, th_h)
        maze.state = init_state

    def search_state(self, maze, s, d, last_actions):
        if d == 0 or maze.state.done != -1:
            if maze.state.done != -1:
                print(maze.state.done, d)
                print(last_actions)
                # maze.show_world()
            return True, maze.state.done
        state = copy.deepcopy(maze.state)
        self.s_map[s] = {}
        # a_stopped = True if last_action is not None and last_action[1] == 4 else False
        for a in maze.possible_action():
            if self.is_inv_action(last_actions[-1][0], a[0]) or \
                    self.is_inv_action(last_actions[-1][1], a[1]):
                continue
            maze.state = copy.deepcopy(state)
            maze.move_ah(*a)
            self.s_count += 1
            self.s_map[s][a] = (self.s_count, -1)
            end, done = self.search_state(maze, self.s_count, d - 1, last_actions + [a])
            if end:
                self.s_count -= 1
                self.s_map[s][a] = (None, done)
        if len(self.s_map[s]) == 0:
            return True, maze.state.done
        return False, -1

    def is_inv_action(self, a1, a2):
        return a1 + a2 == 3

    def _set_tro(self):
        self.t[:, :, -1, -1] = 1
        self.r[:, :, :, :, :] = -1
        self.r[:, :, -1, :, :] = 0
        for s, nexts in self.s_map.items():
            for a_h, a_r in itertools.product(range(4), range(4)):
                if (a_h, a_r) in nexts:
                    n_s, done = nexts[(a_h, a_r)]
                    if n_s is not None:
                        self.t[a_r, a_h, s, n_s] = 1
                        # self.r = np.zeros((self.a_r, self.a_h, self.s, self.th_r, self.th_h))
                    else:
                        self.t[a_r, a_h, s, -1] = 1
                        if done != -1:
                            self.r[a_r, a_h, s, done % 10, done // 10] = 100
                else:
                    self.t[a_r, a_h, s, -1] = 1
                    self.r[a_r, a_h, s, :, :] = -1000

    # def make_single_policy(self):
    #     self.single_t = np.zeros((self.a_r * self.a_h, self.s, self.s))
    #     self.single_r = np.zeros((self.th_r * self.th_h, self.a_r * self.a_h, self.s, self.s))
    #     for a_r in range(self.a_r):
    #         for a_h in range(self.a_h):
    #             self.single_t[self.a_h * a_r + a_h] = self.t[a_r, a_h]
    #             for th_r in range(self.th_r):
    #                 for th_h in range(self.th_h):
    #                     for s in range(self.s):
    #                         self.single_r[self.th_h * th_r + th_h, self.a_h * a_r + a_h, s, :]\
    #                             = self.r[a_r, a_h, s, th_r, th_h]
    #
    #
    #     self.single_q = np.zeros((self.single_r.shape[0], self.a_r, self.s))
    #     for r in range(self.single_r.shape[0]):
    #         q = do_value_iteration(self.single_t, self.single_r[r])
    #         for a_r in range(self.a_r):
    #             self.single_q[r, a_r] = np.max(q[a_r * self.a_h:(a_r + 1) * self.a_h], axis=0)
    #     # print("test")

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
            self.maze.state = copy.deepcopy(state)
            ns = np.argmax(self.t[a_r, a_h, s]) if a_r != -1 else s
            v = np.array([[irl.value_a(ns, th_r, a_r, b) for a_r in range(self.a_r)]
                          for th_r in range(self.th_r)])
            n_a_r = np.argmax(np.max(v, axis=0))
            if ns == self.s - 1:
                self.maze.move_only_a(a_h)
            else:
                self.maze.move_ah(a_h, n_a_r)
            pos[len(pos)] = self._s_data(self.maze.state)
            t[pos_s][a_h] = len(pos) - 1
            if ns != self.s - 1:
                a_h_list_2 = set()
                for a in self.s_map[ns].keys():
                    if a[1] == n_a_r:
                        a_h_list_2.add(a[0])
                self.search_state_for_scinario(ns, len(pos) - 1, list(a_h_list_2),
                                               n_a_r, pos, t, irl, d - 1)

            # print(n_a_r)
            # exit()
            #
            # ns

            # v = np.array([irl.value_a(0, 1, a_r, b[i]) for i in range(len(b))])

            # irl.a_vector_a[s]
            # print(a_h)
        # exit()
        # print(self.t)
        # if d == 0 or maze.state.done != -1:
        #     if maze.state.done != -1:
        #         print(maze.state.done, d)
        #         print(last_actions)
        #         # maze.show_world()
        #     return True, maze.state.done
        # state = copy.deepcopy(maze.state)
        # self.s_map[s] = {}
        # # a_stopped = True if last_action is not None and last_action[1] == 4 else False
        # for a in maze.possible_action():
        #     if self.is_inv_action(last_actions[-1][0], a[0]) or \
        #             self.is_inv_action(last_actions[-1][1], a[1]):
        #         continue
        #     maze.state = copy.deepcopy(state)
        #     maze.move_ah(*a)
        #     self.s_count += 1
        #     self.s_map[s][a] = (self.s_count, -1)
        #     end, done = self.search_state(maze, self.s_count, d - 1, last_actions + [a])
        #     if end:
        #         self.s_count -= 1
        #         self.s_map[s][a] = (None, done)
        # if len(self.s_map[s]) == 0:
        #     return True, maze.state.done
        # return False, -1
