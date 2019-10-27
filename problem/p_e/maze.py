import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import heapq

# a_map = {0: "^", 1: "<", 2: ">", 3: "v"}
a_dir = {0: np.array([-1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]), 3: np.array([1, 0])}


class Node(object):
    def __init__(self, yi, xi):
        self.pos = (yi, xi)

    def add_nexts(self, nodes):
        self.nexts = {}
        for a, d in a_dir.items():
            t = tuple(np.array(self.pos) + d)
            if t in nodes:
                self.nexts[a] = nodes[t]
        self.is_t = len(self.nexts) == 3

    def add_to_next_t(self):
        self.to_next_t = []
        for a, n in self.nexts.items():
            prev_n = self
            mid_n_list = [n.pos]
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
                mid_n_list.append(n.pos)

    def add_to_all_t(self):
        if self.is_t:
            self.to_all_t = {self.pos: 0}
        else:
            self.to_all_t = {}
        q = [(d, n) for _, d, _, n in self.to_next_t]
        heapq.heapify(q)
        while len(q) > 0:
            v, n = heapq.heappop(q)
            m_v = self.to_all_t.get(n.pos, 1000)
            if v < m_v:
                self.to_all_t[n.pos] = v
                for _, d, _, nn in n.to_next_t:
                    heapq.heappush(q, (v + d, nn))

    def __lt__(ob1, ob2):
        if ob1.pos[0] == ob2.pos[0]:
            return ob1.pos[1] < ob2.pos[1]
        return ob1.pos[0] < ob2.pos[0]


class State(object):
    def __init__(self, human, agent, enemys):
        self.human = human
        self.agent = agent
        self.enemys = enemys
        self.done = -1


class Maze(object):
    def __init__(self, file_name):
        self.map = np.array([[int(i) for i in l.rstrip()] for l in open(file_name, "r")])
        self.enemy_num = np.max(self.map) - 4
        self._make_maze_data()

    def _make_maze_data(self):
        self.nodes = {}
        enemys = [None] * self.enemy_num
        for yi, xi in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
            if self.map[yi, xi] > 0:
                self.nodes[(yi, xi)] = Node(yi, xi)
            if self.map[yi, xi] == 2:
                human = (yi, xi)
            if self.map[yi, xi] == 3:
                agent = (yi, xi)
            if self.map[yi, xi] >= 5:
                enemys[self.map[yi, xi] - 5] = (yi, xi)
        for node in self.nodes.values():
            node.add_nexts(self.nodes)
        for node in self.nodes.values():
            node.add_to_next_t()
        for node in self.nodes.values():
            node.add_to_all_t()
        self.state = State(human, agent, enemys)

    def move(self, h_action, a_action):
        self.state.human = self.nodes[self.state.human].nexts[h_action].pos
        if self.state.human in self.state.enemys:
            self.state.done = self.state.enemys.index(self.state.human)
            return
        for ei in range(len(self.state.enemys)):
            self.state.enemys[ei] = self._escape(self.state.enemys[ei])
        self.state.agent = self.nodes[self.state.agent].nexts[a_action].pos
        if self.state.agent in self.state.enemys:
            self.state.done = self.state.enemys.index(self.state.agent)

    def _escape(self, pos):
        node = self.nodes[pos]
        a_list = sorted([(self._min_dist(n, d, mid_n), a) for a, d, mid_n, n in node.to_next_t],
                        reverse=True)
        for _, a in a_list:
            nn = node.nexts[a]
            if not tuple(nn.pos) in self.state.enemys:
                return nn.pos
        return pos

    def _min_dist(self, node, dist, mid_node):
        if tuple(self.state.human) in mid_node:
            # print(self.human.to_all_t[tuple(node.pos)], dist)
            return -1000 + dist - self.nodes[self.state.human].to_all_t[node.pos]
        if tuple(self.state.agent) in mid_node:
            return -1000 + dist - self.nodes[self.state.agent].to_all_t[node.pos] - 0.1
        human, agent = self.nodes[self.state.human], self.nodes[self.state.agent]
        return min(human.to_all_t[node.pos], agent.to_all_t[node.pos])

    def possible_action(self):
        return [i for i in
                itertools.product(self.nodes[self.state.human].nexts.keys(),
                                  self.nodes[self.state.agent].nexts.keys())]

    def show_world(self):
        for yi, xi in itertools.product(range(self.map.shape[0]), range(self.map.shape[1])):
            color = "w" if self.map[yi, xi] > 0 else "brown"
            plt.gca().add_patch(patches.Rectangle((xi, -yi), 1, -1, facecolor=color))
        for e in self.state.enemys:
            plt.gca().add_patch(patches.Rectangle(e[::-1] * np.array([1, -1]), 1, -1,
                                                  facecolor="lightblue"))
        plt.gca().add_patch(patches.Rectangle(self.state.human[::-1] * np.array([1, -1]), 1, -1,
                                              facecolor="r"))
        plt.gca().add_patch(patches.Rectangle(self.state.agent[::-1] * np.array([1, -1]), 1, -1,
                                              facecolor="g"))
        plt.ylim((-self.map.shape[0], 0))
        plt.xlim((0, self.map.shape[1]))
        plt.show()
