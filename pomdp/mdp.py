import numpy as np
import scipy.sparse

class MDP(object):
    def __init__(self, s, a, d=0.99, **kwargs):
        self.s = s
        self.init_states = kwargs["init_states"] if "init_states" in kwargs else (0, )
        self.a = a
        self.d = d
        self.sparse = False

        if "sparse" in kwargs:
            self.sparse = True
            self.t = [scipy.sparse.lil_matrix((self.s, self.s)) for _ in range(self.a)]
            self.r = [scipy.sparse.lil_matrix((self.s, self.s)) for _ in range(self.a)]
            self.q = scipy.sparse.lil_matrix((self.a, self.s))
        else:
            self.t = np.zeros((self.a, self.s, self.s))
            self.r = np.zeros((self.a, self.s, self.s))
            self.q = np.zeros((self.s, self.a))
        self.terminate_states = set()


    def get_next_state(self, action, state):
        return np.random.choice(np.arange(len(self.t[0])), p=self.t[action, state])

    def do_value_iteration(self):
        v = np.zeros(self.t.shape[1])
        n_v = 0
        for _i in range(500):
            n_v = np.sum(self.t * self.r + self.t * v * self.d, axis=2)
            n_v = np.max(n_v, axis=0)
            chk = np.sum(np.abs(n_v - v))
            if chk < 1e-6:
                break
            v = n_v
        self.q = np.sum(self.t * self.r + self.t * n_v * self.d, axis=2)

    def is_terminate(self, s):
        if self.terminate_states is not None and s in self.terminate_states:
            return True
        return False

    def set_terminate_states(self, s):
        self.terminate_states.add(s)
        self.t[:, s, :] = 0
        self.t[:, s, s] = 1

    def pick_initial_state(self):
        return np.random.choice(self.init_states)

    @staticmethod
    def softmax(arr, beta):
        ret = np.exp(arr * beta)
        return ret / np.sum(ret, axis=0)
