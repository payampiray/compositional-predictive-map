import numpy as np
import matplotlib.pyplot as plt


def plot_obj_circle(vec, xy, obj_index=None, radius_obj=None, obj_center=None, size=10,
                    vmax=None, vmin=None):
        c = vec
        x = xy[:, 0]
        y = xy[:, 1]
        if obj_index is not None:
            # c[obj_index] = np.zeros(len(obj_index))
            c = np.delete(c, obj_index)
            x = np.delete(x, obj_index)
            y = np.delete(y, obj_index)

        plt.xlim(-size, size)
        plt.ylim(-size, size)

        plt.scatter(x, y, s=size, c=c, alpha=1, cmap='jet', vmax=vmax, vmin=vmin) #cmap='jet' or 'turbo'
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.colorbar()
        plt.gca().axis('off')

        if obj_center is not None:
            circle = plt.Circle(obj_center, radius_obj, facecolor='white', edgecolor='black')
            plt.gca().add_patch(circle)


def plot_obj_square(vec, xy, obj_index=None, obj_width=None, obj_height=None, size=10, vmax=None):
    c = vec
    x = xy[:, 0]
    y = xy[:, 1]
    if obj_index is not None:
        c = np.delete(c, obj_index)
        x = np.delete(x, obj_index)
        y = np.delete(y, obj_index)

    plt.xlim(-size, size)
    plt.ylim(-size, size)

    plt.scatter(x, y, s=size, c=c, alpha=1, cmap='jet', vmax=vmax)  # cmap='jet' or 'turbo'
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_aspect('equal', adjustable='box')

    if obj_width is not None:
        rect = plt.Rectangle((-obj_width/2, -obj_height/2), obj_width, obj_height, facecolor='white', edgecolor='black')
        plt.gca().add_patch(rect)
    plt.gca().axis('off')


def run_algorithm(env, sample, position):
    N = env.N
    D0 = env.D0

    P, c, lxy, obj_index = env.generate(sample)
    xy = lxy[:, 1:]
    dist = np.sqrt(np.sum((xy - position) ** 2, 1))
    state = np.argmin(dist)

    index = obj_index
    A0 = (env.P0 > 0) + 0.0
    A0[state, index] = 1
    deg = np.sum(A0, axis=1)
    P0 = np.diag(np.reciprocal(deg)) @ A0

    R = P0 - P
    barrier_index = np.flatnonzero(np.sum(np.abs(R), 1))

    C = np.eye(N)
    r = R[barrier_index, :]
    c = C[:, barrier_index]
    Y = -r @ D0 @ c
    A = np.linalg.inv(np.eye(Y.shape[0]) - Y)

    D = -D0 @ c @ A @ r @ D0
    vec = D[state,:]

    computations = {'barrier_index': barrier_index, 'r': r, 'c': c, 'Y': Y, 'A': A, 'R': R, 'C': C,
                    'state': state, 'position': position}

    return vec, obj_index, xy, computations


def get_elements(env, sample, position):
    N = env.N
    D0 = env.D0
    P, c, lxy, obj_index = env.generate(sample)
    xy = lxy[:, 1:]
    dist = np.sqrt(np.sum((xy - position) ** 2, 1))
    state = np.argmin(dist)

    index = obj_index

    A0 = (env.P0 > 0) + 0.0
    A0[state, index] = 1
    deg = np.sum(A0, axis=1)
    P0 = np.diag(np.reciprocal(deg)) @ A0
    R0 = P0 - P
    barrier2states = np.flatnonzero(np.sum(np.abs(R0), 1))

    C = np.eye(N)
    r = R0[barrier2states, :]
    c = C[:, barrier2states]
    Y = -r @ D0 @ c
    A = np.linalg.inv(np.eye(Y.shape[0]) - Y)

    return state, D0, c, r, A, obj_index, xy