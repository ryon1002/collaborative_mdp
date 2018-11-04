import numpy as np


class GraphData(object):
    def __init__(self):
        self.h_node = [["h1a", "h1b"], ["h2a", "h2b"]]
        self.r_node = [["r1a"], ["r2a"]]

        self.h_edge = {
            None: {"h1a": 0, "h1b": 0},
            "h1a": {"h2a": 0},
            "h1b": {"h2b": 0},
        }
        self.r_edge = {
            None: {"r1a": 0},
            "r1a": {"r2a": 0}
        }

        self.cost_candidate = np.array([
            [-5, -10, -30],
            [-5, -30, -10],
        ])

        self.action_index = \
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.items = [
            {"h2a"},
            {"h2b"}
        ]
