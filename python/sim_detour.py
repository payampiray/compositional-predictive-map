import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tools_model as tools


def design(env):
    """Create and return barrier objects."""
    acts1 = 'l0uu0r0'
    wall1 = 'ddllluu'
    start1 = np.array((12, 8))
    object1 = {'start': start1, 'actions': acts1, 'wall': wall1}
    P1 = env.generate(object1)
    goal1 = env.get_trajectory(object1)[1]

    acts2 = 'rrrr0dddd'
    wall2 = 'uuuulllll'
    start2 = np.array((11, 9))
    object2 = {'start': start2, 'actions': acts2, 'wall': wall2}
    P2 = env.generate(object2)
    goal2 = 0

    return P1, P2, goal1, goal2


def main_detour():
    """Main function for detour analysis."""
    n = 20
    env = tools.Barrier(n)
    env.c0 = .01
    DD0 = env.D0
    L0 = np.exp(env.c0) * np.eye(env.N) - env.P0
    D0 = np.linalg.inv(L0)
    error_D0 = np.max(np.abs(D0 - DD0))
    lxy = env.lxy

    P1, P2, goal1, goal2 = design(env)
    goal = goal1

    A1, index_barrier1 = tools.compute_POR(env.P0, D0, P1, goal1)
    A2, index_barrier2 = tools.compute_POR(env.P0, D0, P2, goal2)

    A = sp.linalg.block_diag(A1, A2)
    index_barrier = np.concatenate((index_barrier1, index_barrier2))

    W1 = 1 - (P1 > 0)
    W2 = 1 - (P2 > 0)
    W = np.logical_or(W1, W2) + 0.0

    deg = np.sum(1 - W, axis=1)
    P = np.diag(np.reciprocal(deg)) @ (1 - W)

    start = 0

    fig1 = plt.figure(figsize=(4, 4))
    tools.plot_walls(P1, lxy, linewidth=3, color='green')
    tools.plot_walls(P2, lxy, linewidth=3, color='blue')
    plt.gca().spines[['top']].set_visible(False)

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
    e0 = np.max(np.absolute(D - D_base))

    z_compositional = tools.plan_it(P, D, goal)
    path_compositional = tools.choose_deterministic_path(P, start, goal, np.log(z_compositional+np.finfo(np.float64).eps))

    P1[goal, :] = 0
    P1[goal, goal] = 1

    fig2 = plt.figure(figsize=(4, 4))
    tools.plot_path(path_compositional, lxy)
    tools.plot_walls(P1, lxy, start=start, linewidth=3, color='green')
    tools.plot_walls(P2, lxy, linewidth=3, color='blue')

    return fig1, fig2


def main_plan():
    """Main function for plan analysis."""
    n = 20
    env = tools.Barrier(n)
    env.c0 = .01
    DD0 = env.D0
    L0 = np.exp(env.c0) * np.eye(env.N) - env.P0
    D0 = np.linalg.inv(L0)
    error_D0 = np.max(np.abs(D0 - DD0))
    lxy = env.lxy

    P1, _, goal1, _ = design(env)
    goal = goal1

    A1, index_barrier1 = tools.compute_POR(env.P0, D0, P1, goal1)
    A = A1 + 0.0
    index_barrier = index_barrier1

    W1 = 1 - (P1 > 0)
    W = W1 + 0.0

    deg = np.sum(1 - W, axis=1)
    P = np.diag(np.reciprocal(deg)) @ (1 - W)

    start = 0

    fig1 = plt.figure(figsize=(4, 4))
    tools.plot_walls(P, lxy, linewidth=3, color='green')
    plt.gca().spines[['top']].set_visible(False)

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
    error_distance = np.mean(np.abs(A - A_total))

    D = D0 - D0 @ C[:, index_barrier] @ A @ R[index_barrier, :] @ D0

    L = np.exp(env.c0)*np.eye(env.N) - P
    D_base = np.linalg.inv(L)
    e0 = np.max(np.absolute(D - D_base))

    z_compositional = tools.plan_it(P, D, goal)

    path_compositional = tools.choose_deterministic_path(P, start, goal, np.log(z_compositional+np.finfo(np.float64).eps))

    fig2 = plt.figure(figsize=(4, 4))
    tools.plot_path(path_compositional, lxy)
    tools.plot_walls(P, lxy, start=start, linewidth=3, color='green')

    return fig1, fig2


def main():
    """Main execution function."""
    fig1, fig2 = main_plan()
    fig3, fig4 = main_detour()
    figs = [fig1, fig2, fig3, fig4]
    return figs, 'detour'


if __name__ == "__main__":
    main()
    plt.show()