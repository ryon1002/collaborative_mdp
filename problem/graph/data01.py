import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["h1a", "h1b", "h1c"], ["h2a", "h2b", "h2c"],
                       ["h3a", "h3b", "h3c", "h3d"], ["h4a", "h4b", "h4c", "h4d"]]
        # ["h5a", "h5b", "h5c", "h5d"]]
        # self.r_node = [["r1a", "r1b"], ["r2a", "r2b", "r2c", "r2d"], ["r3a", "r3b", "r3c", "r3d"]]
        self.r_node = [["r1a", "r1b", "r1c"], ["r2a", "r2b", "r2c", "r2d", "r2e", "r2f"],
                       ["r3a", "r3b", "r3c", "r3d", "r3e", "r3f"],
                       ["r4a", "r4b", "r4c", "r4d", "r4e", "r4f"]]
        # ["r5a", "r5b", "r5c", "r5d"]]

        self.h_edge = {
            None: {"h1a": 0, "h1b": 0, "h1c": 0},
            "h1a": {"h2a": 0},
            "h1b": {"h2b": 0},
            "h1c": {"h2c": 0},
            # "h2a": {"h3a": 0},
            # "h2b": {"h3b": 0},
            "h2a": {"h3a": 0},
            "h2b": {"h3b": 0, "h3c": 0},
            "h2c": {"h3d": 0},
            "h3a": {"h4a": 1, "h4b": 2},
            "h3b": {"h4a": 1, "h4b": 2},
            "h3c": {"h4c": 1, "h4d": 2},
            "h3d": {"h4c": 1, "h4d": 2},
            # "h4a": {"h5a": 1},
            # "h4b": {"h5b": 2},
            # "h4c": {"h5c": 2},
            # "h4d": {"h5d": 1},
        }
        self.r_edge = {
            None: {"r1a": 0, "r1b": 0, "r1c": 0},
            "r1a": {"r2a": 0},
            "r1b": {"r2b": 0},
            "r1c": {"r2c": 0, "r2d": 0, "r2e": 0, "r2f": 0},
            "r2a": {"r3a": 1},
            "r2b": {"r3b": 2},
            "r2c": {"r3c": 2},
            "r2d": {"r3d": 1},
            "r2e": {"r3e": 2},
            "r2f": {"r3f": 1},
            "r3a": {"r4a": 1},
            "r3b": {"r4b": 2},
            "r3c": {"r4c": 2},
            "r3d": {"r4d": 1},
            "r3e": {"r4e": 2},
            "r3f": {"r4f": 1},
            # "r3a": {"r4a": 0},
            # "r3b": {"r4b": 2},
            # "r3c": {"r4c": 1},
            # "r3d": {"r4d": 2},
            # "r4a": {"r5a": 1, "r5b": 1},
            # "r4b": {"r5a": 2, "r5b": 2},
            # "r4c": {"r5c": 1, "r5d": 1},
            # "r4d": {"r5c": 2, "r5d": 2},
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
            {"h3a", "h4a", "h3b", "h4b", "r4a", "r4d", "r4e"},
            {"h3c", "h4c", "h3d", "h4d", "r4b", "r4c", "r4f"}
            # {"h5a", "h5c", "r5a", "r5c"},
            # {"h5b", "h5d", "r5b", "r5d"}
        ]
