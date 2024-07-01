import numpy as np
import matplotlib.pyplot as plt
import tools_hoydal as tools
import tools_core as core
from os import path
import pickle as pkl

def run_algorithm(env, obj_index, state_translated):
    N = env.N
    D0 = env.D0

    P = generate_object(env, obj_index)
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


def rotate(xy, obj_index):

    xy_obj = xy[obj_index, :]
    xy_rotated = xy_obj[:, ::-1]
    # xy_rotated = np.zeros((xy_obj.shape[0], 2), dtype=int)
    xy_rotated[:, 0] = -xy_obj[:, 1]
    xy_rotated[:, 1] = +xy_obj[:, 0]

    eps = 10 ** -10
    N = xy.shape[0]
    index_original = np.arange(N)

    index_rotated = np.zeros(xy_rotated.shape[0], dtype=int)
    matched = np.zeros(xy_rotated.shape[0])
    dist_min = np.zeros(xy_rotated.shape[0])
    for i in range(xy_rotated.shape[0]):
        dist = np.sqrt(np.sum((xy_rotated[i, :] - xy[index_original, :])**2, 1))
        j = np.argmin(dist)
        index_rotated[i] = index_original[j]
        dist_min[i] = dist[j]
        matched[i] = np.sum(dist < eps)
        index_original = np.delete(index_original, j)

    return index_rotated


def generate_object(env, object_index):
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


def transform(env, object_transformed, barrier_transformed, state_transformed, POR):
    _, index_obj, xy, computations = run_algorithm(env, object_transformed, state_transformed)
    C = computations['C']
    R = computations['R']
    r = R[barrier_transformed, :]
    c = C[:, barrier_transformed]
    # b1 = np.sort(computations['barrier_index'])
    # b2 = np.sort(barrier_translated)

    D0 = env.D0
    state = computations['state'][0]
    A = POR['A']
    vec = -D0[state] @ c @ A @ r @ D0
    return vec, index_obj, xy


def figure(vec_open, vec_obj1, vec_obj2, index_obj1, index_obj2,
           width_obj1, width_obj2, height_obj1, height_obj2, xy):
    nc = 3
    nr = 1
    size = 10
    obj_index = [None, index_obj1, index_obj2]
    width_obj = [None, width_obj1, width_obj2]
    height_obj = [None, height_obj1, height_obj2]
    vec = [vec_open, vec_obj1, vec_obj2]

    fig = [None] * 1
    fig[0] = plt.figure(figsize=(7, 1.8))
    for j in range(3):
        plt.subplot(nr, nc, j+1)
        tools.plot_obj_square(vec[j], xy, obj_index[j], width_obj[j], height_obj[j], size)

    # plt.colorbar()
    return fig


def main():
    width1 = 8
    height1 = 2
    position1 = np.array((3, 4))

    file_name = path.join('analysis', 'hoydal_rotate.pkl')
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            x = pkl.load(f)
            vec1 = x['vec1']
            vec2 = x['vec2']
            index_obj1 = x['index_obj1']
            index_obj2 = x['index_obj2']
            xy = x['xy']
    else:
        env = core.Hexa_Reg(outline='square')
        sample1 = env.sample(1, 'rect', {'width': width1, 'height': height1})[0]
        # sample2 = env.sample(1, 'circle', {'center': center_obj2, 'radius': radius_obj})[0]

        vec1, index_obj1, xy, computations1 = tools.run_algorithm(env, sample1, position1)
        # vec2, index_obj2, xy, computations2 = tools.run_algorithm(env, sample2, position2)

        barrier_translated = rotate(xy, computations1['barrier_index'])
        object_translated = rotate(xy, index_obj1)
        state_translated = rotate(xy, [computations1['state']])
        vec2, index_obj2, xy = transform(env, object_translated, barrier_translated, state_translated, computations1)
        with open(file_name, 'wb') as f:
            U = {'vec1': vec1, 'index_obj1': index_obj1, 'vec2': vec2, 'index_obj2': index_obj2, 'xy': xy}
            pkl.dump(U, f)

    vec0 = vec1*0
    fig = figure(vec0, vec1, vec2, index_obj1, index_obj2,
           width1, height1, height1, width1, xy)
    fig_name = 'hoydal_rotate'

    return fig, fig_name

