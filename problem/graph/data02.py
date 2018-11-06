import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["h1a", "h1b"], ["h2a", "h2b", "h2c", "h2d"],
                       ["h3a", "h3b", "h3c", "h3d"], ["h4a", "h4b", "h4c", "h4d"],
                       ["h5a", "h5b", "h5c", "h5d"]]
        # self.r_node = [["r1a", "r1b"], ["r2a", "r2b", "r2c", "r2d"], ["r3a", "r3b", "r3c", "r3d"]]
        self.r_node = [["r1a", "r1b"], ["r2a", "r2b", "r2c", "r2d"],
                       ["r3a", "r3b", "r3c", "r3d"], ["r4a", "r4b", "r4c", "r4d"],
                       ["r5a", "r5b", "r5c", "r5d"]]

        self.h_edge = {
            None: {"h1a":0, "h1b":0},
            "h1a": {"h2a": 0, "h2b": 0},
            "h1b": {"h2c": 0, "h2d": 0},
            "h2a": {"h3a": 1},
            "h2b": {"h3b": 2},
            "h2c": {"h3c": 2},
            "h2d": {"h3d": 1},
            "h3a": {"h4a": 1},
            "h3b": {"h4b": 2},
            "h3c": {"h4c": 2},
            "h3d": {"h4d": 1},
            "h4a": {"h5a": 1},
            "h4b": {"h5b": 2},
            "h4c": {"h5c": 2},
            "h4d": {"h5d": 1},
        }
        self.r_edge = {
            None: {"r1a":0, "r1b":0},
            "r1a": {"r2a": 0, "r2b": 0},
            "r1b": {"r2c": 1, "r2d": 0},
            "r2a": {"r3a": 0},
            "r2b": {"r3b": 0},
            "r2c": {"r3c": 1},
            "r2d": {"r3d": 2},
            "r3a": {"r4a": 0},
            "r3b": {"r4b": 2},
            "r3c": {"r4c": 1},
            "r3d": {"r4d": 2},
            "r4a": {"r5a": 1, "r5b": 1},
            "r4b": {"r5a": 2, "r5b": 2},
            "r4c": {"r5c": 1, "r5d": 1},
            "r4d": {"r5c": 2, "r5d": 2},
        }

        self.cost_candidate = np.array([
            [-5, -70, -20],
            [-5, -20, -70],
        ])

        self.action_index = \
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.items = [
            # {"h2a", "h3a", "h3c", "r2a", "r3a", "r2c", "r3c", "r3d"},
            # {"h3b", "h2d", "h3d", "r2a", "r3a", "r2c", "r3c", "r3d"}
            # {"h3a", "h4a", "h4c", "r4a"},
            # {"h4b", "h3d", "h4d", "r4a"}
            {"h5a", "h5c", "r5a", "r5c"},
            {"h5b", "h5d", "r5b", "r5d"}
        ]

