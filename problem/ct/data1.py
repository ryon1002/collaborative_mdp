import numpy as np


class CTData(object):
    def __init__(self):
        # self.color = np.array([[0, 1, 1, 1, 2],
        #                        [2, 1, 2, 3, 3],
        #                        [3, 1, 2, 3, 1],
        #                        [2, 3, 0, 1, 1],
        #                        [1, 1, 2, 2, 2],
        #                        [1, 0, 2, 3, 2]])
        # self.color = np.array([[1, 0, 1, 1, 3],
        #                        [1, 2, 2, 3, 1],
        #                        [1, 3, 1, 1, 1],
        #                        [3, 1, 0, 1, 1],
        #                        [0, 2, 2, 2, 1],
        #                        [1, 1, 3, 2, 1]])
        self.color = np.array([[1, 0, 1, 1, 3],
                               [1, 2, 2, 3, 2],
                               [1, 3, 1, 0, 1],
                               [3, 0, 2, 2, 2],
                               [1, 1, 1, 3, 2]])
        self.shape = self.color.shape
        self.index = 1

        self.bomb = {(0, 3): 0,
                     (2, 0): 1}

        # self.medals = {(3, 1): 0,
        #                (2, 0): 1,
        #                (2, 3): 2,
        #                (1, 4): 3,
        #                (3, 2): 4,
        #                (5, 1): 5}
        self.medals = {(3, 0): 0,
                       (2, 1): 1,
                       (1, 3): 2,
                       (0, 4): 3,
                       (2, 3): 4,
                       (3, 1): 5}


        self.h_chip = np.array([0, 3, 0, 1])
        self.r_chip = np.array([1, 0, 3, 0])

        self.recipe = [[{0, 4}, {2, 5}],
                       [{1, 4}, {3, 5}]]

        self.h_start = np.array([0, 1])
        self.r_start = np.array([4, 3])
