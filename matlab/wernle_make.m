function [task, model] = wernle_make(seed)
if nargin<1
    seed_bottom = 0;
    see_up = 1;
else
    seed_bottom = seed;
    see_up = 50-seed;
end

PcSize = .35;
N = 30;

[xy0, J0, Cv0] = create_env(PcSize, N, 1, seed_bottom);

[xy1, J1, Cv1] = create_env(PcSize, N, 0, see_up);

J = [J0; J1];
xy = [xy0 xy1];
Cv = blkdiag(Cv0, Cv1);


[P0] = get_xy2P(xy0);
[P1] = get_xy2P(xy1);
P0 = blkdiag(P0, P1);
D0 = Cv; %/max(diag(Cv))*exp(cost);

[P] = get_xy2P(xy);

R = P0 - P;
C1 = eye(size(P));

barrier_index = sum(abs(R), 2)>0;
r = R(barrier_index, :);
c = C1(:, barrier_index);

Y = -r*D0*c;
A = (eye(size(Y))-Y)^-1;
U = J - D0 * c * A * r * J;


cellPositions_X = 1:N;
cellPositions_Y = 1:N;
[X, Y] =ndgrid(cellPositions_X,cellPositions_Y); %need for plotting later the grid
xy = [reshape(X,1,[]); reshape(Y,1,[])];
N = size(xy, 2);
lxy = [(1:N)' xy(2,:)' xy(1,:)'];


task = struct('xy', xy, 'P0', P0, 'lxy', lxy, 'J', J, 'D0', D0);
model = struct('r', r, 'c', c, 'A', A, 'Y', Y, 'U', U);

end

function [xy0, J0, Cv0] = create_env(PcSize, N, lower_rect, seed)
cost = .1;

fdir = fullfile('..','analysis', 'base', sprintf('Size%0.2f_Nx%d_Ny%d', PcSize, N, N)); mkdir(fdir);
fname = fullfile(fdir, sprintf('seed%03d.mat', seed));
conf = struct('fname', fname, 'rng_seed', seed, 'PcSize', PcSize, 'paint', 0, 'Nx', N, 'Ny', N, 'step_cost', cost);
f0 = gen_base(conf);

J0 = f0.J;
xy0 = f0.placeCenters;
Cv0 = f0.Cov;

if lower_rect
    idx = xy0(2, :)<5;
else
    idx = xy0(2, :)>5;
end
xy0 = xy0(:, idx);
J0  = J0(idx,:);
Cv0 = Cv0(idx, :);
Cv0 = Cv0(:, idx);

end

% -------------------------------------------------------------------------