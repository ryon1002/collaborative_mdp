import numpy as np
from .import pomdp_base

class POMDP(pomdp_base.POMDP):
    def _pre_calc(self):
        self.update = np.zeros((self.a, self.z, self.s, self.s))
        for s in range(self.s):
            for a in range(self.a):
                self.update[a, :, s] = np.outer(self.o[a, s], self.t[a, s])


