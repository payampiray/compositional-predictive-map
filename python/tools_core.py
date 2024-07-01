from os import path
import pickle as pkl
import numpy as np


def hex_radius(n=50, radius=2, size=10):
    if radius == int(radius):
        id = 'hex_n%d_r%d_size%d' % (n, radius, size)
    else:
        id = 'hex_n%d_r%0.1f_size%d' % (n, radius, size)
    file_name = path.join('analysis', 'P0_%s.pkl' % id)
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            U = pkl.load(f)
            P = U['P']
            lxy = U['lxy']
            id = U['id']
            return P, lxy, id

    size_x = size
    size_y = size
    x1 = np.linspace(-size_x, size_x, int(n * size_x / size_y))
    step = x1[1] - x1[0]
    x2 = x1[:-1] + 0.5*step
    # xx = np.array((x1, x2))
    y1 = np.arange(-size_y, size_y, 2*step*np.sin(np.pi/3))
    y2 = np.arange(-size_y+step*np.sin(np.pi/3), size_y, 2*step*np.sin(np.pi/3))

    xv1, yv1 = np.meshgrid(x1, y1)
    xv2, yv2 = np.meshgrid(x2, y2)
    dots1 = np.zeros((xv1.size, 2))
    dots1[:, 0] = np.reshape(xv1, xv1.size)
    dots1[:, 1] = np.reshape(yv1, xv1.size)
    dots2 = np.zeros((xv2.size, 2))
    dots2[:, 0] = np.reshape(xv2, xv2.size)
    dots2[:, 1] = np.reshape(yv2, xv2.size)
    dots = np.concatenate((dots1, dots2), 0)
    N = dots.shape[0]

    # radius = step*1.005
    A = np.zeros((N, N))
    lxy = np.zeros((N, 3))
    dist_sorted = np.zeros((N, 20))
    for i in range(0, N):
        xy = dots[i, :]
        dist = np.sqrt(np.sum((dots - xy)**2, 1))
        dist[i] = 100*radius
        dist_sorted[i, :] = np.sort(dist)[:20]
        z = np.flatnonzero(dist < radius)
        if len(z) == 0:
            z = np.argmin(dist)
        A[i, z] = 1
        lxy[i, 0] = i
        lxy[i, 1:] = xy

    deg = np.sum(A, axis=1)
    P = np.diag(np.reciprocal(deg))@A

    with open(file_name, 'wb') as f:
        U = {'P': P, 'lxy': lxy, 'id': id,
             'n': n, 'x': size_x, 'y': size_y, 'radius': radius}
        pkl.dump(U, f)

    return P, lxy, id

class Hexa_Reg:
    def __init__(self, outline='circle', n=50, r=2, size=10):
        id = 'hex_%s_n%d_r%d_size%d' % (outline, n, r, size)
        c0 = 0.1
        rg = 2
        file_name = path.join('analysis', 'D0_%s.pkl' % id)
        if path.exists(file_name):
            with open(file_name, 'rb') as f:
                x = pkl.load(f)
                D0 = x['D0']
                P0 = x['P0']
                lxy = x['lxy']
        else:
            if n <= 50:
                size_big = 2 * size
                n_big = 2 * n
            else:
                size_big = 1.2 * size
                n_big = int(1.2 * n)
            P0_big, lxy_big, id_big = hex_radius(n_big, r, size_big)
            if outline == 'circle':
                index = np.sqrt(np.sum(lxy_big[:, 1:] ** 2, 1)) <= size
            elif outline == 'square':
                index = np.logical_and(np.abs(lxy_big[:, 1]) <= size, np.abs(lxy_big[:, 2]) <= size)
            else:
                raise Exception('unknown outline')

            A = (P0_big[index, :] > 0) + 0.0
            A = A[:, index]
            deg = np.sum(A, axis=1)
            P0 = np.diag(np.reciprocal(deg)) @ A
            lxy = np.zeros((np.sum(index), 3))
            lxy[:, 1:] = lxy_big[index, 1:]

            M = P0_big.shape[0]
            D0_big = np.linalg.inv(np.exp(c0) * np.eye(M) - P0_big)
            D0 = D0_big[index, :]
            D0 = D0[:, index]
            with open(file_name, 'wb') as f:
                U = {'P0': P0, 'lxy': lxy, 'D0': D0, 'id': id, 'n': n, 'r': r, 'id_big': id_big}
                pkl.dump(U, f)

        self.name = 'Hexa_Reg'
        self.N = P0.shape[0]
        self.r = r
        self.P0 = P0
        self.id0 = id
        self.lxy = lxy
        self.c0 = c0
        self.D0 = D0
        self.U0 = []
        self.rg = rg

    def sample(self, number_samples, object='circle', properties=None):

        if properties is None:
            properties = {'center': np.array((0, 0)), 'radius': 1}
        samples = [dict() for x in range(number_samples)]
        for i in range(number_samples):
            samples[i] = {'object': object, 'properties': properties}

        return samples

    def generate(self, sample):
        P0 = self.P0
        N = self.N
        lxy = self.lxy
        object = sample['object']
        properties = sample['properties']
        if object == 'circle':
            r_box = properties['radius']
            center = properties['center']
            xy = lxy[:, 1:] - center
            magnitude = np.sqrt(np.sum(xy**2, 1))

            ind_min = np.argmin(magnitude)
            xy = lxy[:, 1:] - lxy[ind_min, 1:]
            magnitude = np.sqrt(np.sum(xy**2, 1))
            index = (magnitude < r_box)
        elif object == 'rect':
            obj_width = properties['width']
            obj_height = properties['height']
            x = lxy[:, 1]
            y = lxy[:, 2]
            index = np.logical_and(np.abs(x) < obj_width/2, np.abs(y) < obj_height/2)
        else:
            raise Exception('Unknown object')

        A = (P0 > 0) + 0.0
        not_index = np.logical_not(index)
        index = np.flatnonzero(index)
        not_index = np.flatnonzero(not_index)
        for i in range(len(index)):
            A[index[i], not_index] = 0
            A[not_index, index[i]] = 0
        deg = np.sum(A, axis=1)
        deg0 = np.flatnonzero(deg == 0)
        for i in range(len(deg0)):
            A[deg0[i], deg0[i]]=1

        deg = np.sum(A, axis=1)
        P = np.diag(np.reciprocal(deg))@A

        c0 = self.c0
        c = c0 * np.ones(N)
        return P, c, lxy, index

    def add_goal(self, center, r_box=1):
        P0 = self.P0
        N = self.N
        lxy = self.lxy
        xy = lxy[:, 1:] - center
        magnitude = np.sqrt(np.sum(xy**2, 1))

        ind_min = np.argmin(magnitude)
        xy = lxy[:, 1:] - lxy[ind_min, 1:]
        magnitude = np.sqrt(np.sum(xy**2, 1))
        index_goal = (magnitude < r_box)

        A = (P0 > 0) + 0.0
        not_index = np.logical_not(index_goal)
        index_goal = np.flatnonzero(index_goal)
        not_index = np.flatnonzero(not_index)
        for i in range(len(index_goal)):
            A[index_goal[i], not_index] = 0
            A[index_goal[i], index_goal[i]] = 1
        deg = np.sum(A, axis=1)
        P = np.diag(np.reciprocal(deg))@A

        c0 = self.c0
        c = c0 * np.ones(N)
        return P, c, lxy, index_goal

