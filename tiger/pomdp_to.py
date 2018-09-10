import numpy as np
from model.pomdp_to import POMDP


class Tiger(POMDP):
    def __init__(self):
        super().__init__(3, 3, 2)
        self.t[:, :, -1] = 1
        self.t[-1, :, :] = np.identity(3)

        self.r[0, :2] = [10, -100]
        self.r[1, :2] = [-100, 10]
        self.r[-1, :2] = -1

        self.o[:2, :, :] = 0.5
        self.o[-1, 0, :] = [0.8, 0.2]
        self.o[-1, 1, :] = [0.2, 0.8]
        self.o[-1, -1, :] = 0.5

        self.pre_calc()
