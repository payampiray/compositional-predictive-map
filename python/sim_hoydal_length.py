import numpy as np
import matplotlib.pyplot as plt
import tools_hoydal as tools
import tools_core as core
from os import path
import pickle as pkl


def get_fieldsize(vecs, xy0, thr):
    K = len(vecs)
    sizes = np.zeros(K)
    for j in range(K):
        vec = vecs[j]
        # thr = np.mean(vec) + 3*np.std(vec)
        z = np.zeros(len(vec))
        z[vec > thr] = 1
        xy = xy0[vec > thr, :]
        dist = np.zeros(xy.shape[0])
        for i in range(xy.shape[0]):
            d = xy - xy[i, :]
            dist[i] = np.max(np.sqrt(np.sum(d**2, 1)))
        # size = np.max(dist)
        sizes[j] = np.mean(z)*400

    return sizes


def figure_circle(vec, obj_index, obj_radius, obj_center, xy, vmax, field_size):
    size = 10
    num_samples = len(vec)
    nc = num_samples
    nr = 1

    fig = [None] * 2
    fig[0] = plt.figure(figsize=(8.5, 1.8))
    for j in range(1, num_samples):
        plt.subplot(nr, nc, j+1)
        tools.plot_obj_circle(vec[j], xy, obj_index[j], obj_radius[j], obj_center, size, vmax)
        # plt.colorbar()

    # fig[1] = plt.figure(figsize=(3, 1.8))
    # j = -1
    # tools.plot_obj_circle(vec[j], xy, obj_index[j], obj_radius[j], obj_center, size, vmax)
    # plt.colorbar()

    font_small = 9
    font_large = 11

    fig[1] = plt.figure(figsize=(2.75, 2.25))
    ax = fig[1].add_axes([.25, .2, .7, .7])
    ax.spines[['right', 'top']].set_visible(False)
    plt.rc('font', size=font_small)
    plt.bar(np.arange(num_samples), field_size,  width=0.5)
    # plt.ylim((150, None))
    plt.xticks(np.arange(num_samples), obj_radius*2)
    plt.xlabel('Object diameter', fontsize=font_large)
    plt.ylabel('Field size', fontsize=font_large)

    return fig


def main_circle():
    obj_center = np.array((0, 1))
    position0 = np.array((-0.5, -3))
    num_samples = 5
    obj_radius = np.array((2, 5, 10, 15, 20))/80*20/2
    displacement_y = -0.5
    n = 100
    file_name = path.join('analysis', 'hoydal_length_n%d.pkl' % n)
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            file = pkl.load(f)
            vectors = file['vectors']
            obj_index = file['obj_index']
            obj_radius = file['obj_radius']
            obj_center = file['obj_center']
            xy = file['xy']
            # displacement_y = file['displacement_yisplacement_y']
            # position0 = file['position0']
    else:
        max = np.zeros(num_samples)
        std = np.zeros(num_samples)
        vectors = [None] * num_samples
        obj_index = [None] * num_samples
        env = core.Hexa_Reg(outline='square', n=n)
        for j in range(0, num_samples):
            position = position0 + np.array((0, displacement_y*j))
            sample = env.sample(1, 'circle', {'center': obj_center, 'radius': obj_radius[j]})[0]
            vectors[j], obj_index[j], xy, _ = tools.run_algorithm(env, sample, position)
            max[j] = np.nanmax(vectors[j])
            std[j] = np.nanstd(vectors[j])

        with open(file_name, 'wb') as f:
            file = {'vectors': vectors, 'obj_index': obj_index, 'obj_radius': obj_radius, 'obj_center': obj_center,
                    'xy': xy, 'displacement_y': displacement_y, 'position0': position0}
            pkl.dump(file, f)

    med = np.median(vectors[2])
    thr = 0.00005
    field_size = get_fieldsize(vectors, xy, med)
    vmax = 0.0025
    fig = figure_circle(vectors, obj_index, obj_radius, obj_center, xy, vmax, field_size)

    return fig


def main():
    fig = main_circle()
    fig_name = 'hoydal_length'
    return fig, fig_name
