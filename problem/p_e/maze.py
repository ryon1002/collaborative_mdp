import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import heapq

# a_map = {0: "^", 1: "<", 2: ">", 3: "v"}
a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class Node(object):
    def __init__(self, yi, xi):
        self.pos = np.array([yi, xi])

    def add_nexts(self, nodes):
        self.nexts = {}
        for a, d in a_dir.items():
            t = tuple(self.pos + d)
            if t in nodes:
                self.nexts[a] = nodes[t]
        self.is_t = len(self.nexts) == 3

    def add_to_next_t(self):
        self.to_next_t = []
        for a, n in self.nexts.items():
            prev_n = self
            mid_n_list = [tuple(n.pos)]
            for d in range(1000):
                if n.is_t:
                    self.to_next_t.append((a, d + 1, mid_n_list, n))
                    break
                to_n = []
                for na, nn in n.nexts.items():
                    if not np.array_equal(nn.pos, prev_n.pos):
                        to_n.append(nn)
                if len(to_n) != 1:
                    print("Error!!")
                    exit()
                prev_n = n
                n = to_n[0]
                mid_n_list.append(tuple(n.pos))

    def add_to_all_t(self):
        if self.is_t:
            self.to_all_t = {tuple(self.pos): 0}
        else:
            self.to_all_t = {}
        q = [(d, n) for _, d, _, n in self.to_next_t]
        heapq.heapify(q)
        while len(q) > 0:
            v, n = heapq.heappop(q)
            p = tuple(n.pos)
            m_v = self.to_all_t.get(p, 1000)
            if v < m_v:
                self.to_all_t[p] = v
                for _, d, _, nn in n.to_next_t:
                    heapq.heappush(q, (v + d, nn))

    def __lt__(ob1, ob2):
        if ob1.pos[0] == ob2.pos[0]:
            return ob1.pos[1] < ob2.pos[1]
        return ob1.pos[0] < ob2.pos[0]


class Maze(object):
    def __init__(self, file_name):
        self.map = np.array([[int(i) for i in l.rstrip()] for l in open(file_name, "r")])
        self._make_maze_data()

    def _make_maze_data(self):
        self.nodes = {}
        self.enemys = [None] * 4
        self.enemys_pos = [None] * 4
        for yi, xi in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
            if self.map[yi, xi] > 0:
                self.nodes[(yi, xi)] = Node(yi, xi)
            if self.map[yi, xi] == 2:
                self.human = self.nodes[(yi, xi)]
            if self.map[yi, xi] == 3:
                self.agent = self.nodes[(yi, xi)]
            if self.map[yi, xi] >= 5:
                self.enemys[self.map[yi, xi] - 5] = self.nodes[(yi, xi)]
        for node in self.nodes.values():
            node.add_nexts(self.nodes)
        for node in self.nodes.values():
            node.add_to_next_t()
        for node in self.nodes.values():
            node.add_to_all_t()
        for ei in range(len(self.enemys)):
            self.enemys_pos[ei] = tuple(self.enemys[ei].pos)

    def _move(self, h_action, a_action):
        self.human = self.human.nexts[h_action]
        for ei in range(len(self.enemys)):
            self.enemys[ei] = self._escape(self.enemys[ei])
            self.enemys_pos[ei] = tuple(self.enemys[ei].pos)
            # self.enemys = [self._escape(e) for e in self.enemys]
        self.agent = self.agent.nexts[a_action]

    def _escape(self, node):
        a_list = sorted([(self._min_dist(n, d, mid_n), a) for a, d, mid_n, n in node.to_next_t],
                        reverse=True)
        for _, a in a_list:
            nn = node.nexts[a]
            if not tuple(nn.pos) in self.enemys_pos:
                return nn

    def _min_dist(self, node, dist, mid_node):
        if tuple(self.human.pos) in mid_node:
            # print(self.human.to_all_t[tuple(node.pos)], dist)
            return -1000 + dist - self.human.to_all_t[tuple(node.pos)]
        if tuple(self.agent.pos) in mid_node:
            return -1000 + dist - self.agent.to_all_t[tuple(node.pos)] - 0.1
        t = tuple(node.pos)
        return min(self.human.to_all_t[t], self.agent.to_all_t[t])

    def show_world(self):
        for yi, xi in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
            color = "w" if self.map[yi, xi] > 0 else "brown"
            plt.gca().add_patch(patches.Rectangle((xi, -yi), 1, -1, facecolor=color))
        plt.gca().add_patch(patches.Rectangle(self.human.pos[::-1] * np.array([1, -1]), 1, -1,
                                              facecolor="r"))
        plt.gca().add_patch(patches.Rectangle(self.agent.pos[::-1] * np.array([1, -1]), 1, -1,
                                              facecolor="g"))
        for e in self.enemys:
            plt.gca().add_patch(patches.Rectangle(e.pos[::-1] * np.array([1, -1]), 1, -1,
                                                  facecolor="lightblue"))
        plt.ylim((-self.map.shape[0], 0))
        plt.xlim((0, self.map.shape[1]))
        plt.show()
