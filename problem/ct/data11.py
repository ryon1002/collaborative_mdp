import numpy as np


class CTData(object):
    def __init__(self):
        self.color = np.array([[0, 1, 1, 3],
                               [1, 2, 0, 2],
                               [1, 0, 1, 2],
                               [3, 2, 2, 3]])
        self.shape = self.color.shape

        self.bomb = {(0, 2): 0,
                     (2, 0): 1}

        self.medals = {(0, 3): 0,
                       (3, 0): 1,
                       (1, 2): 2,
                       (2, 1): 3}

        self.h_chip = np.array([0, 2, 0, 1])
        self.r_chip = np.array([1, 0, 2, 0])

        self.recipe = [[{0, 2}, {1, 3}], [{0, 3}]]

        self.h_start = np.array([0, 0])
        self.r_start = np.array([3, 3])
