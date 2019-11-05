import matplotlib.pyplot as plt
import numpy as np

from algo.double_coop_irl_2 import CoopIRL
from algo.vi import do_value_iteration
from problem.p_e.maze import Maze
from problem.p_e.maze_mdp import MazeMDP


def make_belief():
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    return np.concatenate(([b1], [b2]), axis=0).T


if __name__ == '__main__':
    algo, target, main_th_r = 1, 0, 0
    # algo, target, main_th_r = 2, 1, 0
    map_index = 2
    limit = 15
    # limit = 12
    use_dump = True
    save_dump = False
    maze = Maze(f"problem/p_e/map_data/map{map_index}")
    # maze.move_h(2)
    # actions = [(2, -1), (0, 0), (0, 0), (0, 0), (2, 1), (2, 1), (2, 0), (0, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
    # for a in actions[1:7]:
    #     maze.move_ah(*a)
    # maze.show_world()
    # maze.move_ah(2, 1)
    # maze.show_world()
    # exit()

    env = MazeMDP(maze, limit)
    # exit()
    env.make_single_policy()
    irl = CoopIRL()
    irl.calc_h_belief(env, env.single_q)
    # exit()

    # env.a_vector_a, env.h_pi = pickle.load(open("policy.pkl", "rb"))
    # env.make_scinario(0, 1, algo)
    # scinario = worst2.make_worst(2, env)
    # exit()

    # b = make_belief()
    b = np.array([1.0])

    # ii = 0
    # if ii == 0:
    #     beliefs = {}
    #     for th_r in range(env.th_r):
    #         beliefs[th_r] = {}
    #         for s in range(env.s):
    #             beliefs[th_r][s] = np.array([0.5, 0.5])
    # else:
    #     beliefs = env.calc_belief()
    for d in [limit]:
        # env.calc_a_vector(d, beliefs, 1)
        irl.calc_a_vector(env, d, b, algo, use_dump, save_dump)
        irl.calc_belief(env)
    env.make_scinario(f"pv_data/scinario_{map_index}.json", irl)
    # exit()

    # env.make_scinario(main_th_r, index, algo, target)
    # scinario = worst2.make_worst(index, 0, env)
    # pickle.dump((env.a_vector_a, env.h_pi), open("policy.pkl", "wb"))

    for a_r in range(env.a_r):
        v = np.array([irl.value_a(0, 0, a_r, b[i]) for i in range(len(b))])
        # v = np.array([irl.value(0, 1, b[i]) for i in range(len(b))])
        if np.max(v) < -900:
            continue
        # plt.plot(b[:, 0], v, label=a_r)
        plt.plot([1.0], v, label=a_r)
        plt.legend()
        print(a_r, v)
    # plt.show()

    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(11, 0, a_r, b[i]) for i in range(len(b))])
    #     if np.max(v) < -999:
    #         continue
    #     plt.plot(b[:, 0], v, label=a_r)
    #     plt.legend()
    # plt.show()
    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(42, 0, a_r, b[i]) for i in range(len(b))])
    #     if np.max(v) < -999:
    #         continue
    #     plt.plot(b[:, 0], v, label=a_r)
    #     plt.legend()
    # plt.show()
