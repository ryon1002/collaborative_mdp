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
    map_index = 101
    # map_index = 14
    limit = 15
    # limit = 1a
    use_dump = False
    save_dump = False
    maze = Maze(f"problem/p_e/map_data/map{map_index}")
    maze.show_world()
    # exit()

    # # # actions = [(0, -1), (1, 0), (1, 0), (1, 1), (1, 0), (1, 0), (1, 0), (3, 0), (3, 0), (3, 1),
    # # #            (3, 1), (3, 0), (3, 0), (3, 0), (3, 1)]
    # actions = [(2, -1)] + [(2, 1)] * 9
    # check_limit = len(actions) - 1
    # # check_limit = 5
    # # # for ai in range(check_limit):
    # # #     maze.move_ha(actions[ai][0], actions[ai + 1][1])
    # #     # maze.show_world()
    # maze.move_h(actions[0][0])
    # for ai in range(1, check_limit):
    #     maze.move_ah(actions[ai][0], actions[ai][1])
    # #     maze.show_world()
    # maze.show_world()
    # exit()

    env = MazeMDP(maze, limit)
    # exit()
    s_limit = env.sd + 1
    # exit()
    env.make_single_policy()
    print(env.s, s_limit)
    irl = CoopIRL()
    irl.calc_h_belief(env, env.single_q, 0.1)
    print(env.single_q[:, :, :, 0])
    # exit()

    # env.a_vector_a, env.h_pi = pickle.load(open("policy.pkl", "rb"))
    # env.make_scinario(0, 1, algo)
    # scinario = worst2.make_worst(2, env)
    # exit()

    if env.th_h == 1:
        b = np.array([1.0])
    elif env.th_h == 2:
        b = make_belief()

    irl.calc_a_vector(env, s_limit, b, algo, use_dump, save_dump)
    # env.make_scinario(f"pv_data/scinario_101.json", irl, limit, 5, -1)
    env.make_scinario(f"pv_data/scinario_101.json", irl, limit, 5, -1)
    # env.make_scinario(f"pv_data/scinario_{map_index}.json", irl, limit, 5, 0)
    # exit()

    # env.make_scinario(main_th_r, index, algo, target)
    # scinario = worst2.make_worst(index, 0, env)
    # pickle.dump((env.a_vector_a, env.h_pi), open("policy.pkl", "wb"))

    if env.th_h == 2:
        s_belief = [0.5, 0.5]
    elif env.th_h == 1:
        s_belief = [1.0]
    for th_r in range(env.th_r):
        for a_r in range(env.a_r):
            # v = np.array([irl.value_a(0, th_r, a_r, b[i]) for i in range(len(b))])
            v = np.array(irl.value_a(0, th_r, a_r, s_belief))
            # v = np.array([irl.value(0, 1, b[i]) for i in range(len(b))])
            if np.max(v) < -900:
                continue
            # plt.plot(b[:, 0], v, label=a_r)
            # plt.plot([1.0], v, label=a_r)
            # plt.legend()
            print(th_r, a_r, v)
    # plt.show()

    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(11, 0, a_r, b[i]) for i in range(len(b))])
    #     if np.max(v) < -999:
    #         continue
    #     plt.plot(b[:, 0], v, label=a_r)
    #     #     plt.legend()
    # plt.show()
    # for a_r in range(env.a_r):
    #     v = np.array([env.value_a(42, 0, a_r, b[i]) for i in range(len(b))])
    #     if np.max(v) < -999:
    #         continue
    #     plt.plot(b[:, 0], v, label=a_r)
    #     plt.legend()
    # plt.show()
