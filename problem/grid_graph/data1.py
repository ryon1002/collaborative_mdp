import numpy as np
import itertools
# import networkx
import time
from algo.prob_util import exp_prob


class GraphData(object):
    def __init__(self, item_dist, agent_dist):
        self.item_num = len(item_dist)
        self.item_dist = item_dist
        self.agent_dist = agent_dist

        # self.item_dist = np.array([[0, 6, 5, 9, 10, 15],
        #                            [6, 0, 9, 7, 16, 13],
        #                            [5, 9, 0, 4, 7, 10],
        #                            [9, 7, 4, 0, 9, 6],
        #                            [10, 16, 7, 9, 0, 5],
        #                            [15, 13, 10, 6, 5, 0]])
        # self.agent_dist = np.array([[9, 11, 4, 4, 5, 6],
        #                             [7, 13, 4, 6, 3, 8]])
        # self.agent_dist = np.array([[7, 13, 4, 6, 3, 8],
        #                             [9, 11, 4, 4, 5, 6]])

        # self.item_dist = np.array([[0, 29, 20, 11, 25, 18],
        #                            [29, 0, 27, 23, 23, 32],
        #                            [20, 27, 0, 16, 27, 27],
        #                            [11, 23, 16, 0, 21, 20],
        #                            [25, 23, 27, 21, 0, 19],
        #                            [18, 32, 27, 20, 19, 0]])
        # self.agent_dist = np.array([[7, 13, 4, 19, 11, 13],
        #                             [10, 11, 11, 9, 3, 1]])

        # factor = np.random.randint(3, 20, self.item_dist ** 2) \
        #     .reshape((self.item_dist, self.item_dist))
        # self.item_dist = (factor + factor.T) * (1 - np.identity(self.item_dist))
        # self.agent_dist = np.random.randint(1, 20, 2 * self.item_dist).reshape((2, self.item_dist))

    def check_data(self):
        # start = time.time()
        min_cost = []
        min_cost_with_i = []
        for i in range(self.item_num):
            cost_list = self.calc_cost_list_with_a(i)
            min_cost.append(np.min(cost_list))
            min_cost_with_i.append(np.sum(np.array(cost_list) * exp_prob(-np.array(cost_list))))
            print(np.min(cost_list),
                  np.sum(np.array(cost_list) * exp_prob(-np.array(cost_list))))
        min_items = np.where(np.array(min_cost) == np.min(min_cost))[0]
        min_items_with_i = np.argmin(min_cost_with_i)
        # exit()

        check = (min_items_with_i in min_items)
        if not check:
            check = (min_cost[min_items_with_i] - np.min(min_cost) <= 0)

        if not check:
            # print("tt")
            # print(min_cost_with_i, min_items)
            # print(min_cost, min_cost_with_i, min_items_with_i)
            # print(min_cost[min_items_with_i] - np.min(min_cost))
            pass
        return check
        # print(np.argmin(min_cost_with_i))
        # print(time.time() - start)

    def calc_min_cost(self):
        min = 1000
        for perm in itertools.permutations(range(6)):
            prem = list(perm)
            for thred in range(1, 6):
                t = self.calc_cost(perm[:thred], prem[thred:])
                if t < min:
                    min = t
        return min

    def calc_cost_list_with_a(self, agent_a):
        cost_list = []
        item_set = set([i for i in range(self.item_num) if i != agent_a])
        for perm in itertools.permutations(item_set):
            perm = [agent_a] + list(perm)
            for thred in range(1, len(perm)):
                t = self.calc_cost(perm[:thred], perm[thred:])
                cost_list.append(t)
        return cost_list
        # return min(cost_list), cost_list
        # # return min(cost_list), exp_prob(-np.array(cost_list))
        # # return min(cost_list), np.max(exp_prob(-np.array(cost_list)))
        # return np.sum(np.array(cost_list) * exp_prob(-np.array(cost_list)))

    def calc_cost(self, a_items, h_items):
        return max(self.calc_one_cost(a_items, 0), self.calc_one_cost(h_items, 1))

    def calc_one_cost(self, items, index):
        return self.agent_dist[index][items[0]] + \
               sum([self.item_dist[items[i]][items[i + 1]] for i in range(len(items) - 1)])
