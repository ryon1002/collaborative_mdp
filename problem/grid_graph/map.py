import numpy as np


class ItemMap(object):
    def __init__(self, diff_matrix=None):
        # self.items = np.array([(1, 6),
        #                        (2, 1),
        #                        (6, 2),
        #                        (7, 5),
        #                        (12, 1),
        #                        (13, 5)])

        # diff_matrix = np.array([[-8, 4],
        #                         [-7, -1],
        #                         [-3, 0],
        #                         [-2, 3],
        #                         [3, -1],
        #                         [4, 3]])

        self.center = np.array([9, 9])
        # self.a_h = np.array([(9, 8), (9, 10)])
        # self.items = self.center + diff_matrix
        self.a_h = np.array([(3, 5), (3, 7)])
        self.items = np.array(
            [[0, 0],
             [1, 12],
             [2, 0],
             [4, 6],
             [5, 0],
             [6, 7]]
        )

    def make_matrix(self):
        l = len(self.items)
        item_dist = np.zeros((l, l), dtype=np.int)
        for i in range(l):
            for j in range(i + 1, l):
                item_dist[i, j] = np.sum(np.abs(self.items[i] - self.items[j]))
        item_dist += item_dist.T
        agent_dist = np.array([np.sum(np.abs(self.items - i), 1) for i in self.a_h])
        return item_dist, agent_dist

        print(item_dist)
