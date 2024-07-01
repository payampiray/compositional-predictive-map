import numpy as np
import matplotlib.pyplot as plt
import tools_hoydal as tools
import tools_core as core
from os import path
import pickle as pkl


def figure(vecs, obj_index, obj_centers, radius_obj, xy):
    size = 10
    vec = np.amax(vecs, 0)

    # fig = [None] * 1
    # fig[0] = plt.figure(figsize=(7, 1.8))
    tools.plot_obj_circle(vec, xy, obj_index, None, None, size)
    for j in range(len(obj_centers)):
        circle = plt.Circle(obj_centers[j], radius_obj, facecolor='white', edgecolor='black')
        plt.gca().add_patch(circle)
        plt.gca().axis('off')

    # plt.colorbar()


def run_algorithm(env, samples, obj_centers, orient, dist):
    dx = dist*np.cos(np.deg2rad(orient))
    dy = dist*np.sin(np.deg2rad(orient))

    num_samples = len(samples)
    vec = np.zeros((num_samples, env.N))
    obj_index = np.zeros(0, dtype=int)
    # obj_centers = [None] * num_samples
    A = [None] * num_samples
    # for j in range(num_samples):
    for j in range(num_samples):
        sample = samples[j]
        center = sample['properties']['center']
        position = center + np.array((dx, dy))
        state, D0, c, r, A[j], object, xy = tools.get_elements(env, sample, position)
        # if j > 0:
        #     A[j] = A[0]
        D1 = -(D0 @ c) @ A[j] @ r @ D0
        vec[j, :] = D1[state, :]
        obj_index = np.concatenate((obj_index, object))

    return vec, obj_index, obj_centers, xy

def run_save(outline, file_name, dist, orient, radius_obj, obj_centers):
    file_name = path.join('analysis', '%s.pkl' % file_name)
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            x = pkl.load(f)
            vec = x['vec']
            obj_index = x['obj_index']
            obj_centers = x['obj_centers']
            xy = x['xy']
    else:
        env = core.Hexa_Reg(outline=outline)
        samples = []
        for j in range(len(obj_centers)):
            samples.append(env.sample(1, 'circle', {'center': obj_centers[j], 'radius': radius_obj})[0])
        vec, obj_index, obj_centers, xy = run_algorithm(env, samples, obj_centers, orient, dist)
        with open(file_name, 'wb') as f:
            U = {'vec': vec, 'obj_index': obj_index, 'obj_centers': obj_centers, 'xy': xy}
            pkl.dump(U, f)

    return vec, obj_index, obj_centers, xy


def main_1():
    radius_obj = 1
    center_obj1 = np.array((0, 7))
    center_obj2 = np.array((-3, -4))
    center_obj3 = np.array((+6, -1))
    orient = -60
    dist = 3
    outline = 'circle'

    obj_centers = [center_obj1, center_obj2, center_obj3]
    file_name = 'hoydal_multi1'
    vec, obj_index, obj_centers, xy = run_save(outline, file_name, dist, orient, radius_obj, obj_centers)
    figure(vec, obj_index, obj_centers, radius_obj, xy)


def main_2():
    radius_obj = 1
    obj_centers = []
    obj_centers.append(np.array((1, 6)))
    obj_centers.append(np.array((-4, 2)))
    obj_centers.append(np.array((+7, 2)))
    obj_centers.append(np.array((1, -1)))
    obj_centers.append(np.array((-4, -5)))
    obj_centers.append(np.array((+7, -5)))

    outline = 'circle'
    orient = +165
    dist = 3
    file_name = 'hoydal_multi2'
    vec, obj_index, obj_centers, xy = run_save(outline, file_name, dist, orient, radius_obj, obj_centers)
    figure(vec, obj_index, obj_centers, radius_obj, xy)



def main_3():
    radius_obj = 1
    center_obj1 = np.array((0, 8))
    center_obj2 = np.array((-7, -1))
    center_obj3 = np.array((+7, +1))
    orient = -90
    dist = 6
    outline = 'square'

    obj_centers = [center_obj1, center_obj2, center_obj3]
    file_name = 'hoydal_multi3'
    vec, obj_index, obj_centers, xy = run_save(outline, file_name, dist, orient, radius_obj, obj_centers)
    figure(vec, obj_index, obj_centers, radius_obj, xy)

def main():
    nr = 1
    nc = 3
    fig = plt.figure(figsize=(6.5, 1.8))

    plt.subplot(nr, nc, 1)
    main_1()
    plt.subplot(nr, nc, 2)
    main_2()
    plt.subplot(nr, nc, 3)
    main_3()

    fig = [fig]
    fig_name = 'hoydal_multi'
    return fig, fig_name


