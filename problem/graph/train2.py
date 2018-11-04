import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["h1a", "h1b"], ["h2a", "h2b", "h2c", "h2d"]]
        self.r_node = [["r1a", "r1b"], ["r2a", "r2b", "r2c", "r2d"]]

        self.h_edge = {
            None: {"h1a": 0, "h1b": 0},
            "h1a": {"h2a": 1, "h2b": 2},
            "h1b": {"h2c": 1, "h2d": 2},
        }
        self.r_edge = {
            None: {"r1a": 0, "r1b": 0},
            "r1a": {"r2a": 1, "r2b": 1},
            "r1b": {"r2c": 2, "r2d": 2},
        }

        self.cost_candidate = np.array([
            [-5, -10, -30],
            [-5, -30, -10],
        ])

        self.action_index = \
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.items = [
            {"h2a", "h2b", "r2a", "r2c"},
            {"h2c", "h2d", "r2b", "r2d"}
        ]
