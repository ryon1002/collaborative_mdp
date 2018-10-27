import numpy as np


class BuildData(object):
    def __init__(self):
        # self.h_node = [["a", "b", "c", "d"], ["e", "f"]]
        # self.r_node = [["A", "C", "D"], ["E", "F"]]
        self.h_node = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
        self.r_node = [["A", "C", "D"], ["E", "F", "G", "H"]]

        self.action_index =\
            {a: n for n, a in enumerate(sum(self.h_node, []) + sum(self.r_node, []))}

        self.recipe = [[
            ["a", "e", "A", "E"],
            ["a", "f", "A", "F"],
            ["a", "e", "C", "E"],
            ["a", "f", "D", "F"],
            ["c", "e", "A", "E"],
            ["d", "f", "A", "F"],
            ["c", "e", "C", "E"],
            ["d", "f", "D", "F"],
        ], [
            ["b", "g", "A", "G"],
            ["b", "h", "A", "H"],
            ["b", "g", "C", "G"],
            ["b", "h", "D", "H"],
            ["d", "g", "A", "G"],
            ["c", "h", "A", "H"],
            ["d", "g", "C", "G"],
            ["c", "h", "D", "H"],
        ]]

        # self.recipe = [[
        #     ["a", "e", "A", "E"],
        #     ["a", "f", "A", "F"],
        #     ["a", "e", "C", "E"],
        #     ["a", "f", "D", "F"],
        #     ["c", "e", "A", "E"],
        #     ["d", "f", "A", "F"],
        #     ["c", "e", "C", "E"],
        #     ["d", "f", "D", "F"],
        # ], [
        #     ["b", "h", "A", "G"],
        #     ["b", "g", "A", "H"],
        #     ["b", "h", "C", "G"],
        #     ["b", "g", "D", "H"],
        #     ["d", "h", "A", "G"],
        #     ["c", "g", "A", "H"],
        #     ["d", "h", "C", "G"],
        #     ["c", "g", "D", "H"],
        # ]]

        self.recipe_set = []
        for r in self.recipe:
            self.recipe_set.append({tuple(sorted(re)) for re in r})

        self.cost_candidate = np.array([
            [-20, -20, -10, -10, -30, -10, -30, -10, -10, -12, -12, -10, -10, -10, -10],
            [-20, -20, -10, -10, -10, -30, -10, -30, -10, -12, -12, -10, -10, -10, -10],
            # [-20, -20, -10, -10, -20, -20, -20, -20, -10, -12, -12, -10, -10, -10, -10],
            # [-20, -20, -10, -10, -20, -20, -20, -20, -10, -12, -12, -10, -10, -10, -10],
        ])
