import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tools_model as tools


def analyze_barriers(env, object1, object2, num_samples=9):
    """
    Analyze barriers and compute learning results. Combines the functionality
    of err_distance, compute_POR, and learn functions.
    """
    start1 = object1['start'].copy()
    start2 = object2['start'].copy()
    lxy = env.lxy
    N = env.P0.shape[0]
    C = np.eye(N)

    # Initialize arrays
    error_distance = np.zeros(num_samples)
    distance_matrix = np.zeros(num_samples, dtype=int)
    P_matrix = [None] * num_samples

    # Analyze different distances
    for i in range(num_samples):
        # Update positions
        if np.mod(i, 2) == 0:
            start1 = start1 + np.array((0, 1))
        else:
            start2 = start2 - np.array((0, 1))
        distance_matrix[i] = (start2[1]) - (start1[1] + 1)

        # Generate matrices for current positions
        object1['start'], object2['start'] = start1, start2
        P1 = env.generate(object1)
        P2 = env.generate(object2)

        # Compute POR elements for both barriers
        R1 = env.P0 - P1
        R2 = env.P0 - P2
        index_barrier1 = np.flatnonzero(np.sum(np.abs(R1), 1))
        index_barrier2 = np.flatnonzero(np.sum(np.abs(R2), 1))

        Y1 = -R1[index_barrier1, :] @ env.D0 @ C[:, index_barrier1]
        Y2 = -R2[index_barrier2, :] @ env.D0 @ C[:, index_barrier2]
        A1 = np.linalg.inv(np.eye(Y1.shape[0]) - Y1)
        A2 = np.linalg.inv(np.eye(Y2.shape[0]) - Y2)
        A = sp.linalg.block_diag(A1, A2)

        # Calculate combined wall effects
        W = np.logical_or(1 - (P1 > 0), 1 - (P2 > 0)) + 0.0
        deg = np.sum(1 - W, axis=1)
        P = np.diag(np.reciprocal(deg)) @ (1 - W)
        P_matrix[i] = P

        # Compute final error
        R = env.P0 - P
        index_barrier = np.concatenate((index_barrier1, index_barrier2))
        Y = -R[index_barrier, :] @ env.D0 @ C[:, index_barrier]
        A_total = np.linalg.inv(np.eye(Y.shape[0]) - Y)
        error_distance[i] = np.mean(np.abs(A - A_total))

    # Compute learning error using last configuration
    error_learning = compute_learning_error(env.P0, env.D0, P_matrix[-1], A1, A2, index_barrier)

    return P_matrix[0], P_matrix[-1], lxy, distance_matrix, error_distance, error_learning


def compute_learning_error(P0, D0, P, A1, A2, index_barrier, alpha=0.2, num_iterations=100):
    """Compute learning error progression."""
    N = P0.shape[0]
    R = P0 - P
    C = np.eye(N)
    Y = -R[index_barrier, :] @ D0 @ C[:, index_barrier]
    A_total = np.linalg.inv(np.eye(Y.shape[0]) - Y)

    A = sp.linalg.block_diag(A1, A2)
    I = np.eye(A.shape[0])
    errors = np.zeros(num_iterations)

    for i in range(num_iterations):
        A = A + alpha * (I + A @ Y - A)
        errors[i] = np.mean(np.abs(A - A_total))

    return errors


def visualize(P_both1, P_both2, lxy, distance, error_distance, error_learning):
    font_huge = 24
    font_large = 12
    font_small = 12
    linewidth = 2
    wspace = .45
    hspace = .25
    bottom = 0.2
    nr = 1
    nc1 = 3
    nc2 = 2
    figsize1 = (16, 4.5)
    figsize2 = (8, 3)

    num_samples = len(error_distance)

    fig1 = plt.figure(figsize=figsize1)
    # plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)

    plt.subplot(nr, nc1, 1)
    tools.plot_walls(P_both1, lxy, linewidth=linewidth)
    plt.gca().spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    #
    # plt.subplot(nr, nc1, 2)
    # tools.plot_walls(P2, lxy, linewidth=linewidth)
    # plt.gca().spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    plt.subplot(nr, nc1, 2)
    tools.plot_walls(P_both1, lxy, linewidth=linewidth)
    plt.title('Distance = 8', fontsize=font_huge)

    plt.subplot(nr, nc1, 3)
    tools.plot_walls(P_both2, lxy, linewidth=linewidth)
    plt.title('Distance = 0', fontsize=font_huge)

    plt.show()
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


def main():
    """Main execution function combining main_learn and main."""
    # Initialize environment and objects
    n = 20
    env = tools.Barrier(n)

    # Define objects
    object1 = {
        'start': np.array((11, 16)),
        'actions': 'rr0ddd0ll0u00r00',
        'wall': 'uuurrrrdddlluuur'
    }
    object2 = {
        'start': np.array((11, 6)),
        'actions': 'lll00rrr',
        'wall': 'uuuulddd'
    }

    # Run analysis
    P_first, P_last, lxy, distance_matrix, error_distance, error_learning = \
        analyze_barriers(env, object2, object1)

    # Create and return visualizations
    figs = visualize(P_first, P_last, lxy,
                     distance_matrix, error_distance, error_learning)

    return figs, 'learn'

if __name__ == "__main__":
    main()
    plt.show()