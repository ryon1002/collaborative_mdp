import numpy as np
from .import pomdp_base

class POMDP(pomdp_base.POMDP):
    def _pre_calc(self):
        self.update = np.zeros((self.a, self.z, self.s, self.s))
        for s in range(self.s):
            for a in range(self.a):
                p_z_as = np.dot(self.t[a, s], self.o[a, :])
                self.update[a, :, s] = np.outer(p_z_as, self.t[a, s])
