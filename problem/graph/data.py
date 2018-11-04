import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["a", "b", "c", "d"], ["e", "f"]]
        self.r_node = [["A", "C", "D"], ["E", "F"]]

        self.h_edge = {
            None: {"a": 1, "b": 1, "c": 0, "d": 0},
            "a": {"e": 1},
            "b": {"f": 1},
            "c": {"e": 2, "f": 3},
            "d": {"e": 3, "f": 2},
        }
        self.r_edge = {
            None: {"A": 0,  "C": 4, "D": 4},
            "A": {"E": 1, "F": 1},
            "C": {"E": 2, "F": 2},
            "D": {"E": 3, "F": 3},
        }

        self.cost_candidate = np.array([
            [-5, -20, -5, -30, -6, -25],
            [-5, -20, -30, -5, -6, -25],
        ])

        self.action_index = \
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.items = [
            {"e", "E"},
            {"f", "F"}
        ]

        # self.recipe_set = []
        # for r in self.recipe:
        #     self.recipe_set.append({tuple(sorted(re)) for re in r})
