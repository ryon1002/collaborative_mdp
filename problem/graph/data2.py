import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["h1a", "h1b"], ["h2a", "h2b", "h2c", "h2d"], ["h3a", "h3b", "h3c", "h3d"]]
        self.r_node = [["r1a", "r1b"], ["r2a", "r2b", "r2c", "r2d"], ["r3a", "r3b", "r3c", "r3d"]]

        self.h_edge = {
            None: {"h1a":0, "h1b":0},
            "h1a": {"h2a": 1, "h2b": 2},
            "h1b": {"h2c": 2, "h2d": 1},
            "h2a": {"h3a": 1},
            "h2b": {"h3b": 2},
            "h2c": {"h3c": 2},
            "h2d": {"h3d": 1},
        }
        self.r_edge = {
            # None: {"r1a":0, "r1b":0},
            # "r1a": {"r2a": 1, "r2b": 2},
            # "r1b": {"r2c": 1, "r2d": 2},
            # "r2a": {"r3a": 1},
            # "r2b": {"r3b": 2},
            # "r2c": {"r3c": 1},
            # "r2d": {"r3d": 2},
            None: {"r1a":0, "r1b":0},
            "r1a": {"r2a": 1, "r2b": 1},
            "r1b": {"r2c": 2, "r2d": 2},
            "r2a": {"r3a": 1},
            "r2b": {"r3b": 1},
            "r2c": {"r3c": 2},
            "r2d": {"r3d": 2},
        }

        self.cost_candidate = np.array([
            [-5, -5, -30],
            [-5, -30, -5],
        ])

        self.action_index = \
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.items = [
            # {"h2a", "h3a", "h3c", "r2a", "r3a", "r2c", "r3c"},
            # {"h3b", "h2d", "h3d", "r3d"}
            {"h2a", "h3a", "h3c"},
            {"h3b", "h2d", "h3d"}
        ]

