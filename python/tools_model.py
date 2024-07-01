import numpy as np
import matplotlib.pyplot as plt
from os import path
import pickle as pkl
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap


def def_grid(n):
    N = int(n**2)
    x = np.linspace(1, n, n)
    y = np.linspace(1, n, n)

    xv, yv = np.meshgrid(x, y)
    dots = np.zeros((N, 2))
    dots[:, 0] = np.reshape(xv, n**2)
    dots[:, 1] = np.reshape(yv, n**2)

    r = 1.000
    A = np.zeros((N, N))
    lxy = np.zeros((N, 3))
    for i in range(N):
        xy = dots[i, :]
        dist = np.sqrt(np.sum((dots - xy)**2, 1))
        A[i, np.flatnonzero(dist == r)] = 1
        lxy[i, 0] = i
        lxy[i, 1:] = xy

    deg = np.sum(A, axis=1)
    P = np.diag(np.reciprocal(deg))@A

    return P, lxy

def plot_walls(P, lxy, linewidth=1, start=None, last=None):
    W = 1-(P>0)

    # terminal is not a wall
    terminals = np.flatnonzero(np.diag(P))
    for t in terminals:
        to_terminals = np.flatnonzero(P[:, t])
        W[t, to_terminals] = 0

    N = np.shape(W)
    N = N[0]
    ptr_down = np.zeros(N)
    ptr_right = np.zeros(N)

    for i in range(N):
        xyi = lxy[i,1:3]
        xyir = xyi+[1, 0]
        j = (lxy[:,1]==xyir[0]) & (lxy[:,2]==xyir[1])
        if any(j):
            j = int(lxy[j,0])
            if W[i,j] != 0:
                ptr_down[i] = 1

        xyir = xyi+[0, 1]
        j = (lxy[:,1]==xyir[0]) & (lxy[:,2]==xyir[1])
        if any(j):
            j = int(lxy[j,0])
            if W[i,j] != 0:
                ptr_right[i] = 1

    rr = lxy[:,1]
    cc = lxy[:,2]
    row = np.max(rr)
    col = np.max(cc)

    # fig = plt.figure()
    # ax = plt.axes()
    # ax = fig.add_axes([0.5, 0.5, rr, cc])

    constant = 0.5
    for i in range(ptr_right.size):
        if ptr_right[i] == 1: # right passage blocked:
            x, y = [cc[i]+constant, cc[i]+constant], [rr[i]-constant, rr[i]+constant]
            plt.plot(x, y, color='black', linewidth=linewidth)

        if ptr_down[i] == 1: # right passage blocked:
            x, y = [cc[i]-constant, cc[i]+constant], [rr[i]+constant, rr[i]+constant]
            plt.plot(x, y, color='black', linewidth=linewidth)

        if any(terminals == i):
            center = [cc[i]-constant, rr[i]-constant]
            obj_width = 1
            obj_height = 1
            rect = plt.Rectangle(center, obj_width, obj_height, facecolor='red',
                                 edgecolor='red', alpha=0.6)
            plt.gca().add_patch(rect)

        if i<1000:
            x, y = [cc[i]+constant, cc[i]+constant], [rr[i]-constant, rr[i]+constant]
            x, y = [cc[i], cc[i]], [rr[i], rr[i]]
            ys = np.mean(y)

            x, y = [cc[i], cc[i]], [rr[i], rr[i]]
            xs = np.mean(x)

            # s = str(i)
            # plt.text(xs, ys, s)

    if start is not None:
        i = start
        center = [cc[i]-constant, rr[i]-constant]
        obj_width = 1
        obj_height = 1
        rect = plt.Rectangle(center, obj_width, obj_height, facecolor='blue',
                             edgecolor='blue', alpha=0.6)
        plt.gca().add_patch(rect)
    if last is not None:
        i = last
        center = [cc[i] - constant, rr[i] - constant]
        obj_width = 1
        obj_height = 1
        rect = plt.Rectangle(center, obj_width, obj_height, facecolor='blue',
                             edgecolor='blue', alpha=1)
        plt.gca().add_patch(rect)


    plt.xlim(-constant + np.min(cc), constant + np.max(cc))
    plt.ylim(-constant + np.min(rr), constant + np.max(rr))

    # plt.xlim(constant, col+constant) #+constant
    # plt.ylim(constant, row+constant) #+constant
    # plt.gca().invert_yaxis()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    # plt.axis('square')
    # max_xy = np.max(lxy[:, 1:], 0)
    # plt.xlim((1-constant, max_xy[0]))
    # plt.ylim((1-constant, max_xy[1]))
    # plt.show()

def plot_path(path, lxy, linewidth=0.5):
    rr = lxy[:, 1]
    cc = lxy[:, 2]

    constant = 0.5
    for i in path:
        center = (cc[i]-constant, rr[i]-constant)
        obj_width = 1
        obj_height = 1
        rect = plt.Rectangle(center, obj_width, obj_height, facecolor='red'
                             , alpha=0.3) #edgecolor='red'
        plt.gca().add_patch(rect)

        if i<1000:
            x, y = [cc[i]+constant, cc[i]+constant], [rr[i]-constant, rr[i]+constant]
            x, y = [cc[i], cc[i]], [rr[i], rr[i]]
            ys = np.mean(y)

            x, y = [cc[i], cc[i]], [rr[i], rr[i]]
            xs = np.mean(x)

            # s = str(i)
            # plt.text(xs, ys, s)

    plt.xlim(constant + np.min(cc), constant + np.max(cc))
    plt.ylim(constant + np.min(rr), constant + np.max(rr))

    # plt.xlim(constant, col+constant) #+constant
    # plt.ylim(constant, row+constant) #+constant
    # plt.gca().invert_yaxis()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.axis('square')
    max_xy = np.max(lxy[:, 1:], 0)
    plt.xlim((1-constant, max_xy[0]+constant))
    plt.ylim((1-constant, max_xy[1]+constant))
    # plt.show()

def object_rotate(P_original, lxy):
    xy = np.transpose(lxy[:, 1:])
    n = np.max(xy, 1)[1]
    R = np.array([[0, 1], [-1, 0]])
    xy_rot = R@xy
    xy_rot[1, :] = xy_rot[1, :] + n+1

    idx_old2new = np.ones(lxy.shape[0], dtype=int)*-1
    for i in range(len(idx_old2new)):
        j = (xy_rot[0,i] == xy[0,:]) & (xy_rot[1,i] == xy[1,:])
        j = np.flatnonzero(j)
        if len(j) != 1:
            raise Exception('error')
        idx_old2new[i] = j

    N = P_original.shape[0]
    P_rot = np.zeros((N, N))
    for i in range(len(idx_old2new)):
        p = P_original[i, :]
        P_rot[idx_old2new[i], idx_old2new] = p

    return P_rot, idx_old2new

def create_object(P0, lxy, start, actions, wall):
        A = (P0 > 0) + 0.0

        N = lxy.shape[0]
        s = start
        len_barrier = len(actions)
        wall_down = np.zeros(N, dtype=bool)
        wall_up = np.zeros(N, dtype=bool)
        wall_right = np.zeros(N, dtype=bool)
        wall_left = np.zeros(N, dtype=bool)
        trajectory = np.array(s)
        for k in range(len_barrier):
            xys = lxy[s, 1:]
            a = actions[k]
            w = wall[k]
            w_u, w_d, w_r, w_l = 0, 0, 0, 0
            if a == 'u':
                dxy = np.array([1, 0])
            elif a == 'r':
                dxy = np.array([0, 1])
            elif a == 'd':
                dxy = np.array([-1, 0])
            elif a == 'l':
                dxy = np.array([0, -1])
            elif a == '0':
                dxy = np.array([0, 0])
            else:
                raise Exception('unknown action')
            if w == 'u':
                dw = np.array([1, 0])
            elif w == 'd':
                dw = np.array([-1, 0])
            elif w == 'r':
                dw = np.array([0, 1])
            elif w == 'l':
                dw = np.array([0, -1])
            elif w == '0':
                dw = None
            else:
                raise Exception('unknown wall!')

            s_old = s
            xy_next = xys + dxy
            s = np.flatnonzero(np.all(lxy[:, 1:] == xy_next, 1))
            if len(s) == 0:
                s = 1
            else:
                s = s[0]
            trajectory = np.append(trajectory, s)
            if dw is not None:
                xy_discont = xys + dw
                s_discont = np.flatnonzero(np.all(lxy[:, 1:] == xy_discont, 1))[0]
                A[s_old, s_discont] = 0
                A[s_discont, s_old] = 0
                # if not np.any(trajectory == s_old):
                #     trajectory = np.append(trajectory, s_old)
                # if not np.any(trajectory == s_discont):
                #     trajectory = np.append(trajectory, s_discont)

            wall_down[s] = w_d
            wall_up[s] = w_u
            wall_right[s] = w_r
            wall_left[s] = w_l

        deg = np.sum(A, axis=1)
        P = np.diag(np.reciprocal(deg)) @ A
        # plot_walls(P, lxy)

        return P, trajectory


def slice_DR(n, c0=None, m=100):
    if c0 is None:
        c0 = .1
        none_c0 = True
        file_name = path.join('analysis', 'D_n%d.pkl' % n)
    else:
        none_c0 = False
        file_name = path.join('analysis', 'D_n%d_c%0.2f.pkl' % (n, c0))

    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            D = pkl.load(f)
            D_main = D['D0']
            return D_main

    n_original = n
    nn = [20, 25, 40]
    if n_original not in nn:
        nn.append(n_original)

    M = m**2
    P, _ = def_grid(m)
    D_big = np.linalg.inv(np.exp(c0)*np.eye(M) - P)

    for n in nn:
        if none_c0:
            file_name = path.join('analysis', 'D_n%d.pkl' % n)
        else:
            file_name = path.join('analysis', 'D_n%d_c%0.2f.pkl' % (n, c0))

        if not path.exists(file_name):
            if n<m:
                half_m_n = int((m-n)/2)
                s_base = m*half_m_n + half_m_n
                idx = np.zeros(0, dtype=int)
                for k in range(n):
                    ii = np.arange(s_base + k * m, s_base + k * m + n, dtype=int)
                    idx = np.concatenate((idx,ii))
                D_sliced = D_big[idx, :]
                D_sliced = D_sliced[:, idx]
            elif n==m:
                D_sliced = D_big

            D = {'num_row': n, 'num_row_base': m, 'c': c0, 's_init': s_base, 'D0': D_sliced}
            with open(file_name, 'wb') as f:
                pkl.dump(D, f)
            if n == n_original:
                D_main = D_sliced

    return D_main

class Barrier:
    def __init__(self, n):
        P0, lxy = def_grid(n)
        c0 = 0.01
        dim = np.array([n,n])
        N = P0.shape
        N = N[0]

        file_name = path.join('analysis', 'D_n%d_c0.01.pkl' % (n,))
        if path.exists(file_name):
            with open(file_name, 'rb') as f:
                x = pkl.load(f)
                D0 = x['D0']
        else:
            D0 = slice_DR(n, c0)

        self.n = n
        self.N = N
        self.P0 = P0
        self.lxy = lxy
        self.dim = dim
        self.c0 = c0
        self.D0 = D0

    def sample(self, number_samples, kind=None, len_barrier=None, terminals=None):
        N = self.N
        if len_barrier is None:
            len_barrier = self.n

        if terminals is None:
            terminals = np.floor(N * np.random.rand(number_samples))
            terminals = terminals.astype(int)

        random_seeds = 10 ** 6 * np.random.rand(number_samples)
        random_seeds = random_seeds.astype(int)

        samples = [dict() for x in range(number_samples)]
        for i in range(number_samples):
            samples[i] = {'terminal': terminals[i], 'random_seed': random_seeds[i], 'kind': kind,
                          'len_barrier': len_barrier, 'args': []}

        return samples

    def generate(self, sample):
        start = sample['start']
        actions = sample['actions']
        wall = sample['wall']

        P0 = self.P0
        lxy = self.lxy
        # lxy = lxy[:, [0, 2, 1]]

        dist = np.sum((lxy[:, 1:] - start)**2, 1)
        s0 = np.argmin(dist)

        P, trajectory = create_object(P0, lxy, s0, actions, wall)

        return P

    def add_object(self, sample, P0):
        start = sample['start']
        actions = sample['actions']
        wall = sample['wall']

        lxy = self.lxy
        # lxy = lxy[:, [0, 2, 1]]

        dist = np.sum((lxy[:, 1:] - start)**2, 1)
        s0 = np.argmin(dist)

        P, trajectory = create_object(P0, lxy, s0, actions, wall)

        return P, trajectory

    def get_trajectory(self, sample):
        start = sample['start']
        actions = sample['actions']
        wall = sample['wall']

        P0 = self.P0
        lxy = self.lxy
        # lxy = lxy[:, [0, 2, 1]]

        dist = np.sum((lxy[:, 1:] - start)**2, 1)
        s0 = np.argmin(dist)

        P, trajectory = create_object(P0, lxy, s0, actions, wall)

        return trajectory

    def rotate(self, P_original):
        lxy = self.lxy

        P_rot, idx_old2new = object_rotate(P_original, lxy)

        return P_rot, idx_old2new

    def translate(self, sample, dxy):
        start = sample['start']
        actions = sample['actions']
        wall = sample['wall']

        lxy = self.lxy
        P0 = self.P0

        dist = np.sum((lxy[:, 1:] - start)**2, 1)
        s = np.argmin(dist)

        start = start + dxy
        dist = np.sum((lxy[:, 1:] - start)**2, 1)
        s_translated = np.argmin(dist)
        ds = s_translated - s

        xy = lxy[:, 1:]
        xy_translated = xy + dxy
        idx = np.zeros
        idx_old2new = lxy[:, 0].astype(int) + ds
        # idx_old2new = np.ones(lxy.shape[0], dtype=int) * -1
        # for i in range(len(idx_old2new)):
        #     j = (xy_translated[0, i] == xy[0, :]) & (xy_translated[1, i] == xy[1, :])
        #     j = np.flatnonzero(j)
        #     idx_old2new[i] = j

        P, trajectory = create_object(P0, lxy, s_translated, actions, wall)

        c0 = self.c0
        c = c0 * np.ones(self.N)
        return P, idx_old2new
        # return ds, start

# --------------------
# added for the sim_plan


def add_goal(P0, D0, goal):
    P = P0 + 0.0
    P[goal, :] = 0
    P[goal, goal] = 1

    L = np.exp(0.1)*np.eye(P.shape[0]) - P
    D_base = np.linalg.inv(L)

    barrier2states = np.zeros(1, dtype=int)
    barrier2states[0] = goal
    R = P0 - P
    C = np.eye(R.shape[0])
    Y = -R[barrier2states, :] @ D0 @ C[:, barrier2states]
    A = np.linalg.inv(np.eye(Y.shape[0]) - Y)
    D = D0 - D0 @ C[:, barrier2states] @ A @ R[barrier2states, :] @ D0

    er = np.max(np.abs(D - D_base))

    return P, D


def compute_POR(P0, D0, Pb, goal):
    N = P0.shape[0]
    P = Pb + 0.0
    if goal != 0:
        P[goal, :] = 0
        P[goal, goal] = 1
    L = np.exp(.1) * np.eye(N) - P

    N = P0.shape[0]
    R = P0 - P
    C = np.eye(N)
    index_barrier = np.flatnonzero(np.sum(np.abs(R), 1))
    Y = -R[index_barrier, :] @ D0 @ C[:, index_barrier]
    A = np.linalg.inv(np.eye(Y.shape[0]) - Y)

    return A, index_barrier


def update_POR(Y, A, alpha=.3, num_steps=1, A_total=0):
    I = np.eye(A.shape[0])
    err = np.zeros(num_steps)
    for i in range(num_steps):
        A = A + alpha*(I + A@Y - A)
        err[i] = np.mean(np.abs(A - A_total))

    if num_steps == 1:
        err = err[0]
    return A, err

def plan_it(P, D0, goal):
    D = D0 + 0.0

    N = D.shape[0]
    states = np.arange(N)
    states = np.delete(states, goal)
    D_eff = D[states, :]
    D_eff = D_eff[:, states]
    t = P[:, goal]
    t = np.delete(t, goal)
    z = np.zeros(N)
    z[states] = D_eff@t
    z[goal] = 1

    if any(z<0):
        min_z = np.min(z[z>0])
        z[z<0] = min_z

    return z

def choose_path(P, start, goal, z):
    max_steps = 1000
    s = start
    s_pre = -1
    do_it = 1
    steps = 0
    path = np.array(s)
    while do_it:
        next = np.flatnonzero(P[s, :])
        if len(next) > 1:
            next = next[next != s_pre]
        z_next = z[next]
        s_pre = s
        p = z_next/np.sum(z_next)
        if np.sum(p) < 0.999:
            print('')
        s = np.random.choice(next, p=p)
        path = np.append(path, s)
        steps = steps+1
        do_it = (s != goal) and (steps < max_steps)

    return path


def choose_deterministic_path(P, start, goal, z):
    max_steps = 1000
    s = start
    s_pre = -1
    do_it = 1
    steps = 0
    path = np.array(s)
    while do_it:
        next = np.flatnonzero(P[s, :])
        if len(next) > 1:
            next = next[next != s_pre]
        z_next = np.exp(z[next])
        s_pre = s
        j = np.argmax(z_next)
        s = next[j]

        path = np.append(path, s)
        steps = steps+1
        do_it = (s != goal) and (steps < max_steps)

    return path


def walk_and_learn(D0, P, D_first, A, Y, Ro, Co, start, goal):
    z_goal = np.exp(1)
    D = D_first + 0.0
    t = P[:, goal] + 0.0
    t[goal] = 0
    z0 = D0 @ t
    z0[goal] = z_goal
    Rz0 = Ro @ t

    max_steps = 1000
    s = start
    s_pre = -1
    do_it = 1
    steps = 0
    path = np.array(s)
    err = np.zeros(0)
    z_err = np.zeros(0)
    while do_it:
        # zz = plan_it(P, D, goal)
        z = z0 - Co @ A @ Rz0
        z[z < 0] = np.min(z[z > 0])
        z[goal] = z_goal
        # z_err = np.append(z_err, np.max(np.abs(z-zz)))

        next = np.flatnonzero(P[s, :])
        if len(next) > 1:
            next = next[next != s_pre]
        z_next = z[next]
        s_pre = s
        p = z_next/np.sum(z_next)
        if np.sum(p) < 0.999:
            print('')
        s = np.random.choice(next, p=p)
        # j = np.argmax(z_next)
        # s = next[j]
        path = np.append(path, s)
        steps = steps+1
        do_it = (s != goal) and (steps < max_steps)

        A, e = update_POR(Y, A)
        err = np.append(err, e)

    return path, z


def plot_z(P, lxy, z):
    image_method = 0
    rr = lxy[:, 1]
    cc = lxy[:, 2]
    N = lxy.shape[0]
    path = np.arange(N)

    # z = z/np.max(z)
    min_z = - np.log(np.finfo(np.float64).eps)
    z = np.log(z)
    # z = z + min_z
    # z = z/np.max(z)
    b = .2
    z = 1. / (1 + np.exp(-b * z))

    if image_method:
        plt.figure()
        plot_walls(P, lxy, linewidth=2)

        extent = 0.5, 20.5, 0.5, 20.5
        n = int(np.sqrt(P.shape[0]))
        matrix = np.zeros((n, n))
        # for i in range(n):
        #     idx = i + np.arange(0, n**2, n)
        #     matrix[i, :] = z[idx]
        for i in range(z.shape[0]):
            center = (int(cc[i]-1), int(rr[i]-1))
            matrix[n-1-center[1], center[0]] = z[i]

        cmap = pl.cm.Reds
        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))
        # Set alpha
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)

        plt.imshow(matrix, cmap=my_cmap, alpha=matrix,  extent=extent)
        plt.colorbar()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

    # ------------------------------------------------------
    plt.figure()
    plot_walls(P, lxy, linewidth=2)
    constant = 0.5
    for i in path:
        center = (cc[i]-constant, rr[i]-constant)
        obj_width = 1
        obj_height = 1
        rect = plt.Rectangle(center, obj_width, obj_height, facecolor='red',
                             alpha=z[i])
        plt.gca().add_patch(rect)

    plt.xlim(constant + np.min(cc), constant + np.max(cc))
    plt.ylim(constant + np.min(rr), constant + np.max(rr))

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.axis('square')
    max_xy = np.max(lxy[:, 1:], 0)
    plt.xlim((1-constant, max_xy[0]+constant))
    plt.ylim((1-constant, max_xy[1]+constant))

def sr(P, start, goal):
    N = P.shape[0]
    M = np.eye(N)
    gamma = 1
    r = -.1*np.ones(N)
    r[goal] = 1
    beta = 1
    alpha = .2

    max_steps = 10000
    s = start
    s_pre = -1
    do_it = 1
    steps = 0
    path = np.array(s)
    while do_it:
        z = M@r
        z[goal] = r[goal]

        next = np.flatnonzero(P[s, :])
        if len(next) > 1:
            next = next[next != s_pre]
        z_next = z[next]
        s_pre = s
        p = np.exp(beta*z_next)/np.sum(np.exp(beta*z_next))
        s = np.random.choice(next, p=p)
        # j = np.argmax(z_next)
        # s = next[j]
        path = np.append(path, s)
        steps = steps+1
        do_it = (s != goal) and (steps < max_steps)

        one = np.zeros(N)
        one[s_pre] = 1
        M[s_pre, :] = M[s_pre, :] + alpha*(one + gamma*M[s, :] - M[s_pre, :])
    return path

def random_walk(P, start, goal):
    max_steps = 10000
    s = start
    s_pre = -1
    do_it = 1
    steps = 0
    path = np.array(s)
    while do_it:
        next = np.flatnonzero(P[s, :])
        if len(next) > 1:
            next = next[next != s_pre]
        s_pre = s
        p = np.ones(len(next))/len(next)
        if np.sum(p) < 0.999:
            print('')
        s = np.random.choice(next, p=p)
        # j = np.argmax(z_next)
        # s = next[j]
        path = np.append(path, s)
        steps = steps+1
        do_it = (s != goal) and (steps < max_steps)

    return path


def td_learn(D_initial, P, start, goal):
    N = P.shape[0]
    gamma = np.exp(-.1)
    M = 1/gamma*D_initial + 0.0
    alpha = .3
    t = P[:, goal] + 0.0
    t[goal] = 0
    z_goal = np.exp(1)

    max_steps = 5000
    s = start
    s_pre = -1
    do_it = 1
    steps = 0
    path = np.array(s)
    err = np.zeros(0)
    z_err = np.zeros(0)
    while do_it:
        z = gamma*M@t
        z[goal] = z_goal
        z[z < 0] = np.min(z[z > 0])

        next = np.flatnonzero(P[s, :])
        if len(next) > 1:
            next = next[next != s_pre]
        z_next = z[next]
        s_pre = s
        p = 1/len(next)
        u = z_next/np.sum(z_next)
        j = np.random.choice(len(next), p=u)
        s = next[j]
        path = np.append(path, s)
        steps = steps+1
        do_it = (s != goal) and (steps < max_steps)

        one = np.zeros(N)
        one[s_pre] = 1
        M[s_pre, :] = M[s_pre, :] + alpha*(one + p/u[j]*gamma*M[s, :] - M[s_pre, :])
    return path, z


def value_iteration(P, goal):
    N = P.shape[0]
    v = np.zeros(N)
    c = -.1
    v[goal] = 1
    r = c*np.ones(N)
    r[goal] = 1

    s = goal
    states = np.zeros(1, dtype=int)
    remained = np.arange(N)
    states[0] = s
    steps = 0
    while (len(states)<N) and (steps<1000):
        steps = steps+1
        next = np.flatnonzero(P[:, s])
        s = np.random.choice(next)
        if all(states != s):
            states = np.append(states, s)
            remained = remained[remained != s]

    states = np.append(states, remained)
    states = states[states != goal]
    #
    # for j in range(N):
    #     next = np.flatnonzero(P[:, s])
    #     for j in next:
    #         if all(states != j):
    #             states = np.append(states, j)

    do_loop = 1
    err = 0.001
    tol_err = .0001
    max_steps = 5000
    steps = 0
    while (err > tol_err) and (steps < max_steps):
        w = v+0.0
        steps = steps + 1
        for s in states:
            next = np.flatnonzero(P[s, :])
            v[s] = np.max(r[s] + v[next])
        err = np.max(np.abs(v-w))

    return v
