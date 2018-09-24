import itertools
import numpy as np
from model.double_coop_irl_from import CoopIRL

class Correct(CoopIRL):
    def __init__(self, p_kind, p_num, objects):
        s = (p_num + 1) ** p_kind
        self.shape = [p_num + 1] * p_kind
        self.objects = [np.array(o) for o in objects]
        super().__init__(s, p_kind, p_kind, len(objects))

    def _set_tro(self):
        shape_list = [range(i) for i in self.shape]
        action_list = [range(a) for a in [self.a_r, self.a_h]]
        for i in itertools.product(*shape_list):
            for a_r, a_h in itertools.product(*action_list):
                s = np.ravel_multi_index(i, self.shape, "clip")
                items = np.array(i)
                items[a_r] += 1
                items[a_h] += 1
                ns = np.ravel_multi_index(items, self.shape, "clip")
                self.t[a_r, a_h, s, ns] = 1
                for no, o in enumerate(self.objects):
                    if np.min(np.array(i) - o) < 0:
                        if np.min(np.array(items) - o) < 0:
                            # if a_r in [3]
                            # print(a_r, a_h, s, no)
                            # reward = -2,
                            self.r[a_r, a_h, s, no] = -1
                        else:
                            self.r[a_r, a_h, s, no] = 10
        # exit()

        # for i in itertools.product(*shape_list):
        #     s = np.ravel_multi_index(i, self.shape, "clip")
        #     for a_r in range(self.a_r):
        #         items = np.array(i)
        #         items[a_r] += 1
        #         for no, o in enumerate(self.objects):
        #             a_h = np.argmax(items - o < 0)
        #             self.o[no, a_r, s, a_h] = 1
