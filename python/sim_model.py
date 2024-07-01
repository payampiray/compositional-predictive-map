import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tools_model as tools


def learn_barriers(env, samples, num_steps=5000):
    P0 = env.P0
    L0 = np.exp(env.c0)*np.eye(env.N) - env.P0
    if env.D0 is None:
        D0 = np.linalg.inv(L0)
    else:
        D0 = env.D0

    number_samples = len(samples)
    learned = [dict() for _ in range(number_samples)]
    for i in range(number_samples):
        (P, c, _) = env.generate(samples[i])
        expc = np.exp(c)
        L = np.diag(expc) - P

        num_changes = np.flatnonzero(np.sum(np.abs(L-L0),1)>0)
        num_changes = len(num_changes)

        N = D0.shape
        N = N[0]
        states = list(range(0,N))
        s = 0
        R = L - L0
        C = np.eye(N)

        num_barrier_visits = 0
        alf = .3
        barrier2states = np.zeros(0,int)
        barriers = np.zeros(0,int)
        B = np.zeros([0, 0])
        A = np.full([num_changes, num_changes], None)
        converge = []
        for j in range(num_steps):
            dp = P[s,:] - P0[s,:]
            if any(dp != 0):
                num_barrier_visits = num_barrier_visits+1
                if any(barrier2states == s):
                    b = np.flatnonzero(barrier2states == s)
                    b = b[0]
                else:
                    barrier2states = np.append(barrier2states, s)
                    b = len(barriers)
                    barriers = np.append(barriers, b)
                    B = sp.linalg.block_diag(B, 1)

                Y = -R[barrier2states, :] @ D0 @ C[:, barrier2states]
                barrier_size = len(barriers)
                one = np.zeros(barrier_size)
                one[b] = 1
                B[:,b] = B[:,b] + alf*(one + B@Y[:,b] - B[:,b])

                if len(barrier2states) == num_changes:
                    A = np.linalg.inv(np.eye(num_changes) - Y)
                    err1 = np.mean(np.abs(A-B))
                else:
                    err1 = None
                converge.append(err1)
            s = np.random.choice(states, 1, p=P[s,:])[0]
        learned[i] = {'A': A, 'R': R, 'C': C, 'B': B, 'barrier2states': barrier2states, 'barriers': barriers,
                      'converge': converge, 'num_barrier_visits': num_barrier_visits, 'num_steps': num_steps}
    return learned


def plot_matrix(D, C, A, R):
    size = A.shape[0]/D.shape[0]
    total = 0.3
    width = [(1-size)*total, size*total, size*total, (1-size)*total]
    margins = [0.05, 0.03, 0.03, 0]
    matrix = [D, C, A, R]
    plt.figure(figsize=(20, 7), dpi=80)
    for i in range(4):
        plt.gcf().add_axes([.2+np.sum(width[:i])+np.sum(margins[:i]), .2, width[i], .7])
        plt.imshow(matrix[i], cmap='Blues', interpolation='nearest')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)


def compute_POR(P0, D0, P):
    N = P0.shape[0]
    R = P0 - P
    C = np.eye(N)
    index_barrier = np.flatnonzero(np.sum(np.abs(R), 1))
    Y = -R[index_barrier, :] @ D0 @ C[:, index_barrier]
    A = np.linalg.inv(np.eye(Y.shape[0]) - Y)
    Ro = R[index_barrier, :] @ D0
    Co = D0 @ C[:, index_barrier]
    mg = np.max(np.absolute(np.linalg.eig(Y)[0]))
    elements = {'D0': D0, 'Ro': Ro, 'Co': Co, 'A': A, 'index_barrier': index_barrier, 'eig': mg}
    return elements


def transform_representation(n, object, transform='rotation', dxy=None):

    env = tools.Barrier(n)
    lxy = env.lxy
    P = env.generate(object)

    D0 = env.D0

    elements = compute_POR(env.P0, D0, P)
    index_barrier1 = elements['index_barrier']
    A1 = elements['A']

    if transform == 'rotation':
        P_tansformed, index_transformed = env.rotate(P)
    elif transform == 'translation':
        P_tansformed, index_transformed = env.translate(object, dxy)
    else:
        raise Exception('unknown transform')

    R2 = env.P0 - P_tansformed
    C2 = np.eye(env.N)

    index_barrier_rotated = index_transformed[index_barrier1]

    Y2 = -R2[index_barrier_rotated, :] @ env.D0 @ C2[:, index_barrier_rotated]
    A2 = np.linalg.inv(np.eye(Y2.shape[0]) - Y2)
    e1 = np.amax(np.abs(A2 - A1))

    return e1, lxy, P, P_tansformed, elements


def figure_learning(P1, P2, P_both1, P_both2, lxy, distance, error_distance, error_learning):
    font_large = 20
    font_small = 10
    linewidth = 2
    wspace = .45
    hspace = .25
    bottom = 0.2
    nr = 1
    nc1 = 4
    nc2 = 2
    figsize1 = (15, 3)
    figsize2 = (8, 3)

    num_samples = len(error_distance)

    fig1 = plt.figure(figsize=figsize1)

    plt.subplot(nr, nc1, 1)
    tools.plot_walls(P1, lxy, linewidth=linewidth)
    plt.gca().spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    plt.subplot(nr, nc1, 2)
    tools.plot_walls(P2, lxy, linewidth=linewidth)
    plt.gca().spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    plt.subplot(nr, nc1, 3)
    tools.plot_walls(P_both1, lxy, linewidth=linewidth)
    plt.title('Distance = 8', fontsize=font_large)

    plt.subplot(nr, nc1, 4)
    tools.plot_walls(P_both2, lxy, linewidth=linewidth)
    # plt.title('Distance = 8', fontsize=font_large)
    plt.title('Distance = 0', fontsize=font_large)

    fig2 = plt.figure(figsize=figsize2)
    plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)

    plt.subplot(nr, nc2, 1)
    plt.rc('font', size=font_small)
    plt.bar(np.arange(num_samples), error_distance,  width=0.5)
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.xticks(np.arange(num_samples), distance)
    plt.xlabel('Distance between objects', fontsize=font_large)
    plt.ylabel('Error', fontsize=font_large)

    # fig[2] = plt.figure(figsize=(5, 5), dpi=80)
    plt.subplot(nr, nc2, 2)
    plt.plot(error_learning)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.rc('font', size=font_small)
    plt.xlabel('Number of updates', fontsize=font_large)
    plt.ylabel('Error', fontsize=font_large)

    return fig1, fig2


def err_distance(n, object1, object2, num_samples):
    start1 = object1['start']
    start2 = object2['start']

    env = tools.Barrier(n)
    lxy = env.lxy

    mg = np.zeros(num_samples)
    no_matched = np.zeros(num_samples)
    error_distance = np.zeros(num_samples)
    distance_matrix = np.zeros(num_samples, dtype=int)
    P_matrix = [None] * num_samples
    A1_matrix = [None] * num_samples
    A2_matrix = [None] * num_samples
    index_barrier_matrix = [None] * num_samples
    for i in range(len(error_distance)):
        if np.mod(i, 2) == 0:
            start1 = start1 + np.array((0, 1))
        else:
            start2 = start2 - np.array((0, 1))
        distance_matrix[i] = (start2[1]) - (start1[1]+1)

        object1['start'] = start1
        object2['start'] = start2
        P1 = env.generate(object1)
        P2 = env.generate(object2)
        elements1 = compute_POR(env.P0, env.D0, P1)
        elements2 = compute_POR(env.P0, env.D0, P2)
        A1 = elements1['A']
        A2 = elements2['A']
        A = sp.linalg.block_diag(A1, A2)
        index_barrier1 = elements1['index_barrier']
        index_barrier2 = elements2['index_barrier']
        index_barrier = np.concatenate((index_barrier1, index_barrier2))
        # mg1 = elements1['eig']
        # mg2 = elements2['eig']

        W1 = 1 - (P1 > 0)
        W2 = 1 - (P2 > 0)
        W = np.logical_or(W1, W2) + 0.0

        deg = np.sum(1 - W, axis=1)
        P = np.diag(np.reciprocal(deg)) @ (1 - W)
        D0 = env.D0

        N = env.P0.shape[0]
        R = env.P0 - P
        C = np.eye(N)
        index_barrier_x = np.flatnonzero(np.sum(np.abs(R), 1))
        if len(index_barrier) != len(index_barrier_x):
            no_matched[i] = 1
        Y = -R[index_barrier, :] @ D0 @ C[:, index_barrier]
        A_total = np.linalg.inv(np.eye(Y.shape[0]) - Y)
        mg[i] = np.max(np.absolute(np.linalg.eig(Y)[0]))
        error_distance[i] = np.mean(np.abs(A - A_total))

        if i == 0:
            P12 = [P1, P2]

        P_matrix[i] = P
        A1_matrix[i] = A1
        A2_matrix[i] = A2
        index_barrier_matrix[i] = index_barrier

    j = -1
    error_learning = learn(env.P0, env.D0, P_matrix[j], A1_matrix[j], A2_matrix[j], index_barrier_matrix[j])

    return P12, P_matrix[0], P_matrix[-1], lxy, distance_matrix, error_distance, error_learning


def learn(P0, D0, P, A1, A2, index_barrier):
    N = P0.shape[0]
    R = P0 - P
    C = np.eye(N)
    Y = -R[index_barrier, :] @ D0 @ C[:, index_barrier]
    A_total = np.linalg.inv(np.eye(Y.shape[0]) - Y)

    w, _ = np.linalg.eig(Y)
    mag_w = np.absolute(w)

    A = sp.linalg.block_diag(A1, A2)
    I = np.eye(A.shape[0])
    alpha = .2
    err = np.zeros(100)
    for i in range(len(err)):
        A = A + alpha*(I + A@Y - A)
        err[i] = np.mean(np.abs(A - A_total))
    return err


def figure_concept(P, P_trans, P_rot, lxy, Do, Co, A, Ro):
    linewidth = 2
    nr = 1
    nc = 4
    figsize1 = (15, 3)

    fig1 = plot_matrix(Do, Co, A, -Ro)

    fig2 = plt.figure(figsize=figsize1)
    # plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)

    plt.subplot(nr, nc, 1)
    tools.plot_walls(P, lxy, linewidth=linewidth)
    # plt.title('Distance = 8', fontsize=font_large)

    plt.subplot(nr, nc, 2)
    tools.plot_walls(P_trans, lxy, linewidth=linewidth)

    plt.subplot(nr, nc, 3)
    tools.plot_walls(P_rot, lxy, linewidth=linewidth)

    plt.subplot(nr, nc, 4)
    plt.imshow(A, cmap='Blues', interpolation='nearest')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    return fig1, fig2


def main_concept():
    n = 20
    start = np.array((17, 10))
    acts = 'rrr00lll'
    wall = 'uuuurddd'
    object1 = {'start': start, 'actions': acts, 'wall': wall}
    dxy = np.array((-6, 0))
    e_trans, lxy, P, P_trans, _ = transform_representation(n, object1, 'translation', dxy)
    e_rot, lxy, P, P_rot, elements = transform_representation(n, object1, transform='rotation')

    Do = elements['D0']
    Ro = elements['Ro']
    Co = elements['Co']
    A = elements['A']
    fig_y, fig1 = figure_concept(P, P_trans, P_rot, lxy, Do, Co, A, Ro)

    start = np.array((17, 13))
    acts = 'llll0dddd0rrrr0uuu0lll0dd0rr0u0l0d0'
    wall = 'uuuuullllldddddrrrruuuullldddrruulu'
    dxy = np.array((-4, 0))

    object2 = {'start': start, 'actions': acts, 'wall': wall}
    e_trans, lxy, P, P_trans, _ = transform_representation(n, object2, 'translation', dxy)
    e_rot, lxy, P, P_rot, elements = transform_representation(n, object2, transform='rotation')
    Do = elements['D0']
    Ro = elements['Ro']
    Co = elements['Co']
    A = elements['A']
    fig_x, fig2 = figure_concept(P, P_trans, P_rot, lxy, Do, Co, A, Ro)

    return fig1, fig2, fig_y, fig_x

def main_learn():
    n = 20
    # start = np.array((5, 5))
    start = np.array((13, 16))
    acts = 'rrr00lll'
    wall = 'uuuurddd'
    object2 = {'start': start, 'actions': acts, 'wall': wall}

    start = np.array((13, 6))
    acts = 'llll0dddd0rrrr0uuu0lll0dd0rr0u0l0d0'
    wall = 'uuuuullllldddddrrrruuuullldddrruulu'

    num_samples = 9

    object1 = {'start': start, 'actions': acts, 'wall': wall}
    P12, P_first, P_last, lxy, distance_matrix, error_distance, error_learning = err_distance(n, object1, object2, num_samples)

    fig1, fig2 = figure_learning(P12[0], P12[1], P_first, P_last, lxy, distance_matrix, error_distance, error_learning)
    return fig1, fig2


def main():
    fig1, fig2, fig5, fig6 = main_concept()
    fig3, fig4 = main_learn()
    figs = [fig1, fig2, fig3, fig4]
    fig_name = 'model'
    return figs, fig_name

