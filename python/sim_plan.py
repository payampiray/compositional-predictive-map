import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from os import path
import tools_model as tools
import pickle as pkl

def design(env, mode):
    if mode == 1:
        start1 = np.array((17, 7))
        start2 = np.array((17, 15))
        start3 = np.array((6, 6))
        start4 = np.array((5, 17))
    elif mode == 2:
        start1 = np.array((14, 10))
        start2 = np.array((15, 12))
        start3 = np.array((9, 9))
        start4 = np.array((8, 14))
    elif mode == 3:
        start1 = np.array((11, 14))
        start2 = np.array((12, 15))
        start3 = np.array((6, 13))
        start4 = np.array((5, 17))
    else:
        raise Exception('error')

    acts1 = 'lll00rrr'
    wall1 = 'uuuulddd'
    object1 = {'start': start1, 'actions': acts1, 'wall': wall1}
    P1 = env.generate(object1)

    acts2 = 'rr0ddd0ll0u00r00'
    wall2 = 'uuurrrrdddlluuur'
    object2 = {'start': start2, 'actions': acts2, 'wall': wall2}
    P2 = env.generate(object2)

    acts3 = 'l00dduull0dddrrrr'
    wall3 = 'uulll000urrrr0uuu'
    object3 = {'start': start3, 'actions': acts3, 'wall': wall3}
    P3 = env.generate(object3)

    acts4 = 'u0l00ddull0d0rrruu'
    wall4 = 'rruulll00urrr0ddll'
    object4 = {'start': start4, 'actions': acts4, 'wall': wall4}
    P4 = env.generate(object4)
    goal4 = env.get_trajectory(object4)[9]

    goal1 = 0
    goal2 = 0
    goal3 = 0

    return P1, P2, P3, P4, goal1, goal2, goal3, goal4


def main_plan(design_mode):
    num_sim = 5000

    file_name = path.join('analysis', 'plan_design%d_%d.pkl' % (design_mode, num_sim))
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            file = pkl.load(f)
            path_length = file['path_length']
            labels = file['labels']
            P = file['P']
            lxy = file['lxy']
            start = file['start']
    else:
        n = 20
        env = tools.Barrier(n)
        env.c0 = .1
        DD0 = env.D0
        L0 = np.exp(env.c0) * np.eye(env.N) - env.P0
        D0 = np.linalg.inv(L0)
        lxy = env.lxy

        P1, P2, P3, P4, goal1, goal2, goal3, goal4 = design(env, design_mode)
        goal = goal4

        A1, index_barrier1 = tools.compute_POR(env.P0, D0, P1, goal1)
        A2, index_barrier2 = tools.compute_POR(env.P0, D0, P2, goal2)
        A3, index_barrier3 = tools.compute_POR(env.P0, D0, P3, goal3)
        A4, index_barrier4 = tools.compute_POR(env.P0, D0, P4, goal4)

        A = sp.linalg.block_diag(A1, A2)
        A = sp.linalg.block_diag(A, A3)
        A = sp.linalg.block_diag(A, A4)

        index_barrier = np.concatenate((index_barrier1, index_barrier2))
        index_barrier = np.concatenate((index_barrier, index_barrier3))
        index_barrier = np.concatenate((index_barrier, index_barrier4))

        W1 = 1 - (P1 > 0)
        W2 = 1 - (P2 > 0)
        W3 = 1 - (P3 > 0)
        W4 = 1 - (P4 > 0)
        W = np.logical_or(W1, W2) + 0.0
        W = np.logical_or(W, W3) + 0.0
        W = np.logical_or(W, W4) + 0.0

        deg = np.sum(1 - W, axis=1)
        # deg[deg == 0] = 1
        P = np.diag(np.reciprocal(deg)) @ (1 - W)

        # tools.plot_walls(P, lxy)

        P[goal, :] = 0
        P[goal, goal] = 1

        R = env.P0 - P
        C = np.eye(P.shape[0])
        index_barrier_x = np.flatnonzero(np.sum(np.abs(R), 1))
        matched = 1
        if len(index_barrier) != len(index_barrier_x):
            matched = 0
        else:
            if any((np.sort(index_barrier_x) - np.sort(index_barrier)) != 0):
                matched = 0
        if matched == 0:
            raise Exception('Error: environments not matched')

        Y = -R[index_barrier, :] @ D0 @ C[:, index_barrier]
        A_total = np.linalg.inv(np.eye(Y.shape[0]) - Y)
        mg = np.max(np.absolute(np.linalg.eig(Y)[0]))
        error_distance = np.mean(np.abs(A - A_total))

        D = D0 - D0 @ C[:, index_barrier] @ A @ R[index_barrier, :] @ D0
        Ro = R[index_barrier, :] @ D0
        Co = D0 @ C[:, index_barrier]

        L = np.exp(env.c0)*np.eye(env.N) - P
        D_base = np.linalg.inv(L)
        e0 = np.max(np.absolute(D - D_base))  # if D is computed based on A_total, then D = D_base

        z_complete = tools.plan_it(P, D_base, goal)
        z_initial = tools.plan_it(P, D, goal)

        start = 19
        length_complete = np.zeros(num_sim)
        length_initial = np.zeros(num_sim)
        length_update = np.zeros(num_sim)
        length_td = np.zeros(num_sim)
        length_random = np.zeros(num_sim)
        length_sr = np.zeros(num_sim)
        z_update = 0
        z_td = 0
        for i in range(num_sim):
            np.random.seed(i)
            path1 = tools.choose_path(P, start, goal, z_initial)
            length_initial[i] = len(path1)

            np.random.seed(i)
            path2, z2 = tools.walk_and_learn(D0, P, D, A, Y, Ro, Co, start, goal)
            z_update = z_update + z2/num_sim
            length_update[i] = len(path2)

            np.random.seed(i)
            path0 = tools.choose_path(P, start, goal, z_complete)
            length_complete[i] = len(path0)

            np.random.seed(i)
            path3, z3 = tools.td_learn(D, P, start, goal)
            length_td[i] = len(path3)

            np.random.seed(i)
            path4 = tools.random_walk(P, start, goal)
            length_random[i] = len(path4)

            np.random.seed(i)
            path5 = tools.sr(P, start, goal)
            length_sr[i] = len(path5)

            if np.mod(i+1, 100) == 0:
                print('sim %d' % (i+1))

            if i == 0:
                path_initial = path1
                path_update = path2
                path_complete = path0
                path_td = path3

        with open(file_name, 'wb') as f:
            path_length = [length_random, length_sr, length_initial, length_td, length_update, length_complete]
            labels = ['Random', 'SR', 'Initial', 'TD update', 'Object update', 'Complete']
            file = {'env': env, 'P': P, 'lxy': lxy, 'start': start, 'goal': goal,
                    'path_length': path_length, 'labels': labels,
                    'z_complete': z_complete, 'z_initial': z_initial, 'z_update': z_update, 'z_td': z_td,
                    'path_initial': path_initial, 'path_update': path_update,
                    'path_complete': path_complete,
                    'path_td': path_td}
            pkl.dump(file, f)

    return P, lxy, start, path_length, labels


def score(path_length):
    max_length = 1000
    num_sim = len(path_length[0])
    m = np.zeros(len(path_length))
    e = np.zeros(len(path_length))
    n = np.zeros(len(path_length))
    for i in range(len(path_length)):
        x = path_length[i]
        m[i] = np.mean(x)
        e[i] = np.std(x)/np.sqrt(num_sim)
        n[i] = np.mean(x >= max_length)
    return m, e


def figure_main(P1, lxy, m1, e1, xlabel, start):
    font_small = 11
    font_large = 18
    n = len(xlabel)

    fig1 = plt.figure(figsize=(4, 4))
    tools.plot_walls(P1, lxy, linewidth=1, start=start)

    fig2 = plt.figure(figsize=(7, 4))
    plt.bar(np.arange(n), m1, yerr=e1, width=0.3)
    plt.yticks(np.arange(0, 2001, 500), fontsize=font_small)

    plt.xticks(np.arange(n), xlabel, fontsize=font_large)
    plt.ylabel('Number of steps to goal', fontsize=font_large)
    return fig1, fig2


def main():
    P1, lxy, start, path_length1, label = main_plan(1)
    P2, _, _, path_length2, _ = main_plan(3)

    label = label[:3] + label[4:]
    path_length1 = path_length1[:3] + path_length1[4:]
    path_length2 = path_length2[:3] + path_length2[4:]
    label[3] = 'Update'

    m1, e1 = score(path_length1)
    m2, e2 = score(path_length2)

    fig1, fig2 = figure_main(P1, lxy, m1, e1, label, start)
    fig3, fig4 = figure_main(P2, lxy, m2, e2, label, start)
    figs = [fig1, fig2, fig3, fig4]
    fig_name = 'plan'
    return figs, fig_name
