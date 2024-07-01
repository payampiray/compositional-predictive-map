import numpy as np
import matplotlib.pyplot as plt
import tools_hoydal as tools
import tools_core as core
from os import path
import pickle as pkl

def run_algorithm(env, obj_index, state_translated):
    N = env.N
    D0 = env.D0

    P = translate_object(env, obj_index)
    xy = env.lxy[:, 1:]
    state = state_translated

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

    vec = -D0[state, :] @ c @ A @ r @ D0

    computations = {'barrier_index': barrier_index, 'r': r, 'c': c, 'Y': Y, 'A': A, 'R': R, 'C': C, 'state': state}

    return vec, obj_index, xy, computations


def translate(center_obj1, center_obj2, xy, obj_index):
    mag = np.sum((xy - center_obj1)**2, 1)
    ind = np.argmin(mag)
    center_obj1 = xy[ind, :]
    mag = np.sum((xy - center_obj2)**2, 1)
    ind = np.argmin(mag)
    center_obj2 = xy[ind, :]

    xy_obj = xy[obj_index, :]
    dxy = center_obj2 - center_obj1

    eps = 10 ** -10
    N = xy.shape[0]
    xy_translated = xy_obj + dxy
    index_translated = np.zeros(xy_translated.shape[0], dtype=int)
    matched = np.zeros(xy_translated.shape[0])
    for i in range(xy_translated.shape[0]):
        # dist = np.sqrt(np.sum((xy_translated[i, :] - xy[index_original, :])**2, 1))
        # j = np.argmin(dist)
        # index_translated[i] = index_original[j]
        # dist_min[i] = dist[j]
        # matched[i] = np.sum(dist < eps)
        # index_original = np.delete(index_original, j)

        dist = np.sqrt(np.sum((xy_translated[i, :] - xy) ** 2, 1))
        j = np.flatnonzero(dist < eps)
        matched[i] = np.sum(dist < eps)
        index_translated[i] = j

    if np.any(matched != 1):
        raise Exception('Not matched')

    return index_translated


def translate_object(env, object_index):
    A = (env.P0 > 0) + 0.0
    N = env.N
    index = np.zeros(N, dtype=int)
    index[object_index] = 1
    not_index = np.logical_not(index)
    index = np.flatnonzero(index)
    not_index = np.flatnonzero(not_index)
    for i in range(len(index)):
        A[index[i], not_index] = 0
        A[not_index, index[i]] = 0
    deg = np.sum(A, axis=1)
    deg0 = np.flatnonzero(deg == 0)
    for i in range(len(deg0)):
        A[deg0[i], deg0[i]] = 1

    deg = np.sum(A, axis=1)
    P = np.diag(np.reciprocal(deg)) @ A
    return P


def compute_translated(env, object_translated, barrier_translated, state_translated, POR):
    D2, index_obj, xy, computations = run_algorithm(env, object_translated, state_translated)
    C = computations['C']
    R = computations['R']
    r = R[barrier_translated, :]
    c = C[:, barrier_translated]
    # b1 = np.sort(computations['barrier_index'])
    # b2 = np.sort(barrier_translated)

    D0 = env.D0
    state = computations['state'][0]
    A = POR['A']
    vec = -D0[state] @ c @ A @ r @ D0
    return vec, index_obj, xy


def figure(vec_open, vec_obj1, vec_obj2, index_obj1, index_obj2, center_obj1, center_obj2, radius_obj, xy):
    nc = 3
    nr = 1
    size = 10
    obj_index = [None, index_obj1, index_obj2]
    obj_centers = [None, center_obj1, center_obj2]
    vec = [vec_open, vec_obj1, vec_obj2]

    fig = [None] * 1
    fig[0] = plt.figure(figsize=(7, 1.8))
    for j in range(3):
        plt.subplot(nr, nc, j+1)
        tools.plot_obj_circle(vec[j], xy, obj_index[j], radius_obj, obj_centers[j], size)
        # plt.colorbar()
    return fig


def main():
    radius_obj = 1
    center_obj1 = np.array((-3, -1))
    center_obj2 = np.array((1, 0))
    position1 = np.array((-2, 4))

    file_name = path.join('analysis', 'hoydal_translate.pkl')
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            x = pkl.load(f)
            vec0 = x['vec0']
            vec1 = x['vec1']
            vec2 = x['vec2']
            index_obj1 = x['index_obj1']
            index_obj2 = x['index_obj2']
            xy = x['xy']
    else:

        env = core.Hexa_Reg()
        eps = 0.01
        sample0 = env.sample(1, 'circle', {'center': center_obj1, 'radius': eps})[0]
        sample1 = env.sample(1, 'circle', {'center': center_obj1, 'radius': radius_obj})[0]
        # sample2 = env.sample(1, 'circle', {'center': center_obj2, 'radius': radius_obj})[0]

        vec0, _, xy, computations0 = tools.run_algorithm(env, sample0, position1)
        vec1, index_obj1, xy, computations1 = tools.run_algorithm(env, sample1, position1)
        # vec2, index_obj2, xy, computations2 = tools.run_algorithm(env, sample2, position2)

        barrier_translated = translate(center_obj1, center_obj2, xy, computations1['barrier_index'])
        object_translated = translate(center_obj1, center_obj2, xy, index_obj1)
        state_translated = translate(center_obj1, center_obj2, xy, [computations1['state']])
        vec2, index_obj2, xy = compute_translated(env, object_translated, barrier_translated, state_translated, computations1)
        with open(file_name, 'wb') as f:
            U = {'vec0': vec0, 'vec1': vec1, 'index_obj1': index_obj1, 'vec2': vec2, 'index_obj2': index_obj2, 'xy': xy}
            pkl.dump(U, f)

    fig = figure(vec0, vec1, vec2, index_obj1, index_obj2, center_obj1, center_obj2, radius_obj, xy)
    fig_name = 'hoydal_translate'

    return fig, fig_name
