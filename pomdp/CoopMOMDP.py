import numpy as np
import itertools

import momdp
import correct_world

class CoopMOMDP(momdp.MOMDP):
    def __init__(self, ws, th_h, a_h, a_r):
        super().__init__(ws, th_h, a_h, a_r)
        self.th_h = th_h # y for MOMDP
        self.a_h = a_h # z for MOMDP
        self.a_r = a_r # a for MOMDP

    def setHuams(self, world, humans):
        for th_h, human in enumerate(humans):
            for a_r in range(world.a):
                pi = np.dot(world.t[a_r], human.pi)
                self.o[th_h, a_r] = pi
                t = np.zeros((world.s, world.s))
                for x in range(world.s):
                    for a_h in range(self.z):
                        t[x] += pi[x, a_h] * self.tx[a_r, a_h, x]
                self.r[th_h, a_r] = np.sum(t * human.r_filter, axis=1)


class HumanModel(object):
    def __init__(self, world, goal):
        self.pi = np.zeros((world.s, world.a))
        self.goals = []
        self.make_policy_and_goals(world, goal)
        self.t = np.zeros(world.s, world.s)
        self.make_tr(world)

    def make_policy_and_goals(self, world, goal):
        goal = np.array(goal)
        for s in range(world.s):
            state = np.unravel_index(s, world.shape)
            lack = (goal - state) > 0
            if np.sum(lack) == 0:
                self.pi[s, 0] = 1
                self.goals.append(s)
            else:
                self.pi[s, np.argmax(lack)] = 1

    def make_tr(self, world):
        self.r_filter = np.zeros((world.s, world.s))
        non_goals = list(set(range(world.s)).difference(self.goals))
        for ng in non_goals:
            self.r_filter[ng, self.goals] = 5
            self.r_filter[ng, non_goals] = -1

class Chef(CoopMOMDP):
    def __init__(self):
        world = correct_world.CorrectWorld((2, 2, 2))
        humans = [HumanModel(world, (1, 1, 0)), HumanModel(world, (0, 1, 1))]
        super().__init__(world.s, len(humans), world.a, world.a)
        for a_r, a_h in itertools.product(range(world.a), repeat=2):
            self.tx[a_r, a_h] = np.dot(world.t[a_r], world.t[a_h])
        self.ty[:, :] = np.identity(len(humans))

        self.setHuams(world, humans)
        self.pre_calc()

class Chef2(CoopMOMDP):
    def __init__(self):
        world = correct_world.CorrectWorld((3, 3, 3, 3))
        humans = [HumanModel(world, (2, 1, 1, 0)), HumanModel(world, (1, 1, 0, 2))]
        super().__init__(world.s, len(humans), world.a, world.a)
        for a_r, a_h in itertools.product(range(world.a), repeat=2):
            self.tx[a_r, a_h] = np.dot(world.t[a_r], world.t[a_h])
        self.ty[:, :] = np.identity(len(humans))

        self.setHuams(world, humans)
        self.pre_calc()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    b = np.concatenate(([b1], [b2]), axis=0).T
    # problem = Chef()
    problem = Chef2()
    for d in [6]:
        problem.calc_a_vector(d, b, with_a=True)
        for a in range(problem.a):
            v = np.array([problem.value_a(0, a, b[i]) for i in range(len(b))])
            print(v)
            plt.plot(b[:, 0], v)
    plt.show()

