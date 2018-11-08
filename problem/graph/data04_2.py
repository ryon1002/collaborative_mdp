import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["h1a", "h1b"], ["h2a", "h2b", "h2c"],
                       ["h3a", "h3b", "h3c", "h3d"], ["h4a", "h4b", "h4c", "h4d"]]
        # self.r_node = [["r1a", "r1b"], ["r2a", "r2b", "r2c", "r2d"], ["r3a", "r3b", "r3c", "r3d"]]
        self.r_node = [["r1a", "r1b"], ["r2a", "r2b"],
                       ["r3a", "r3b", "r3c", "r3d"], ["r4a", "r4b", "r4c", "r4d"]]

        self.h_edge = {
            None: {"h1a": 0, "h1b": 0},
            "h1a": {"h2a": 0, "h2b": 0},
            "h1b": {"h2c": 0},
            "h2a": {"h3a": 2},
            "h2b": {"h3b": 2},
            "h2c": {"h3c": 1, "h3d": 1},
            "h3a": {"h4a": 2},
            "h3b": {"h4b": 2},
            "h3c": {"h4c": 1},
            "h3d": {"h4d": 1},
        }
        self.r_edge = {
            None: {"r1a": 0, "r1b": 0},
            "r1a": {"r2a": 0},
            "r1b": {"r2b": 0},
            "r2a": {"r3a": 0, "r3b": 0},
            "r2b": {"r3c": 1, "r3d": 1},
            "r3a": {"r4a": 2},
            "r3b": {"r4b": 2},
            "r3c": {"r4c": 1},
            "r3d": {"r4d": 1},
        }

        self.cost_candidate = np.array([
            [0, -50, -10],
            [0, -10, -50],
        ])

        self.action_index = \
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.items = [
            # {"h2a", "h3a", "h3c", "r2a", "r3a", "r2c", "r3c", "r3d"},
            # {"h3b", "h2d", "h3d", "r2a", "r3a", "r2c", "r3c", "r3d"}
            # {"h3a", "h4a", "h4c", "r4a"},
            # {"h4b", "h3d", "h4d", "r4a"}
            {"h4a", "h4c", "r4a", "r4c"},
            {"h4b", "h4d", "r4b", "r4d"}
        ]
