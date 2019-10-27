from model.coop_irl_mdp import CoopIRLMDP
import copy


# self.s = s
# self.a_r = a_r
# self.a_h = a_h
# self.th_r = th_r
# self.th_h = th_h
# self.t = np.zeros((self.a_r, self.a_h, self.s, self.s))
# self.r = np.zeros((self.a_r, self.a_h, self.s, self.th_r, self.th_h))

class MazeMDP(CoopIRLMDP):
    def __init__(self, maze, d):
        self.s_count = 0
        self.search_state(maze, 0, 11)
        super().__init__(len(self.s_map) + 1, 4, 4, 4, 4)
        print(s_count)

    def search_state(self, maze, s, d, last_action=None):
        global s_count
        if d == 0 or maze.state.done != -1:
            # if d == 0:
            if maze.state.done != -1:
                print(maze.state.done, d)
            s_count += 1
            return
        state = copy.deepcopy(maze.state)
        for a in maze.possible_action():
            if last_action is not None:
                if self.is_inv_action(last_action[0], a[0]) or \
                        self.is_inv_action(last_action[1], a[1]):
                    continue
            maze.state = copy.deepcopy(state)
            maze.move(*a)
            self.search_state(maze, d - 1, a)
        s_count += 1

    def is_inv_action(self, a1, a2):
        return a1 + a2 == 3
