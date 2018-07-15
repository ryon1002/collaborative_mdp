import numpy as np
import mdp

class CorrectWorld(mdp.MDP):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(np.prod(self.shape), len(self.shape), 1)
        self._add_transition([0] * self.a)

    def _add_transition(self, s):
        s_s = np.ravel_multi_index(s, self.shape)
        for a in range(self.a):
            n_s = list(s)
            n_s[a] += 1
            try:
                n_s_s = np.ravel_multi_index(n_s, self.shape)
            except ValueError:
                self.t[a, s_s, s_s] = 1
                continue
            self.t[a, s_s, n_s_s] = 1
            self._add_transition(n_s)
        return

# class

if __name__ == '__main__':
    np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)
    world = CorrectWorld((4, 4, 4))
    print(world.t)
    print(world.r)

