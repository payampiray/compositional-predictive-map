import numpy as np
import matplotlib.pyplot as plt
import tools_hoydal as tools
import tools_core as core
from os import path
import pickle as pkl

def plot_columns(X, xy, size, obj_index, square_length, circle_radius):
    K = X.shape[1]
    plt.figure(figsize=(20, 1.5))

    nc = 10
    nr = min(10, 1+K//nc)

    # minx = np.min(X)
    # X = X+abs(minx)
    # maxx = np.max(X)
    # X = X/maxx

    # nr = 2
    # nc = 4

    for j in range(K):
        c = X[:, j]
        x = xy[:, 0]
        y = xy[:, 1]
        if obj_index[j] is not None:
            c = np.delete(c, obj_index[j])
            x = np.delete(x, obj_index[j])
            y = np.delete(y, obj_index[j])

        plt.subplot(nr, nc, j+1)
        if j == 0:
            obj = plt.Rectangle((-square_length, -square_length), square_length * 2, square_length * 2, color='w')
        else:
            obj = plt.Circle((0, 0), circle_radius, color='w')
        plt.gca().add_patch(obj)

        plt.scatter(x, y, s=size, c=1-c, alpha=1, cmap='jet') #cmap='jet' or 'turbo'
        # plt.scatter(0, 0, s=400, c=0, alpha=1, cmap='jet') #cmap='jet'
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)


def figure_object(vec_obj1, vec_obj2, index_obj1, index_obj2,
           width_obj1, height_obj1, obj_radius, xy):
    nc = 2
    nr = 1
    size = 10

    fig = plt.figure(figsize=(4.5, 1.8))
    plt.subplot(nr, nc, 1)
    tools.plot_obj_square(vec_obj1, xy, index_obj1, width_obj1, height_obj1, size)

    plt.subplot(nr, nc, 2)
    tools.plot_obj_circle(vec_obj2, xy, index_obj2, obj_radius, np.array((0, 0)))

    # plt.colorbar()
    return fig

def figure_environment(vec_obj1, vec_obj2, index_obj1, index_obj2, center_obj1, center_obj2, radius_obj, xy1, xy2):
    nc = 2
    nr = 1
    size = 10
    obj_index = [index_obj1, index_obj2]
    obj_centers = [center_obj1, center_obj2]
    vec = [vec_obj1, vec_obj2]
    xy = [xy1, xy2]

    fig = plt.figure(figsize=(4.5, 1.8))
    for j in range(2):
        plt.subplot(nr, nc, j+1)
        tools.plot_obj_circle(vec[j], xy[j], obj_index[j], radius_obj, obj_centers[j], size)

    # plt.colorbar()
    return fig


def generalize_objects():
    obj_width1 = 2
    obj_height1 = 2
    obj_center = np.array((0, 0))
    obj_radius = 1.4
    position = np.array((+5, 5))

    env1 = core.Hexa_Reg(outline='square')
    sample1 = env1.sample(1, 'circle', {'center': obj_center, 'radius': obj_radius})[0]
    # env2 = core.Hexa_Reg(outline='circle')
    sample2 = env1.sample(1, 'rect', {'width': obj_width1, 'height': obj_height1})[0]

    D1, obj_index1, xy, _ = tools.run_algorithm(env1, sample1, position)
    D2, obj_index2, xy, _ = tools.run_algorithm(env1, sample2, position)

    fig = figure_object(D1, D2, obj_index1, obj_index2, obj_width1, obj_height1, obj_radius, xy)
    return fig


def generalize_environments():
    radius_obj = 1
    center_obj1 = np.array((-2, 2))
    center_obj2 = np.array((-5, 5))
    position1 = (-5, 2)
    position2 = (-8, 5)

    file_name = path.join('analysis', 'hoydal_generalize.pkl')
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            x = pkl.load(f)
            vec1 = x['vec1']
            index_obj1 = x['index_obj1']
            xy1 = x['xy1']
            vec2 = x['vec2']
            index_obj2 = x['index_obj2']
            xy2 = x['xy2']
    else:
        env1 = core.Hexa_Reg(outline='circle')
        sample1 = env1.sample(1, 'circle', {'center': center_obj1, 'radius': radius_obj})[0]
        env2 = core.Hexa_Reg(outline='square')
        sample2 = env2.sample(1, 'circle', {'center': center_obj2, 'radius': radius_obj})[0]
        vec1, index_obj1, xy1, _ = tools.run_algorithm(env1, sample1, position1)
        vec2, index_obj2, xy2, _ = tools.run_algorithm(env2, sample2, position2)

        with open(file_name, 'wb') as f:
            U = {'vec1': vec1, 'index_obj1': index_obj1, 'xy1': xy1, 'vec2': vec2, 'index_obj2': index_obj2, 'xy2': xy2}
            pkl.dump(U, f)

    fig = figure_environment(vec1, vec2, index_obj1, index_obj2, center_obj1, center_obj2, radius_obj, xy1, xy2)

    return fig


def main():
    # fig1 = generalize_objects()
    fig2 = generalize_environments()
    fig = [fig2]
    fig_name = 'hoydal_generalize'
    return fig, fig_name
