function [models, turning1, alternating1, turning2, alternating2, J] = derdikman_make(seed)
if nargin<1, seed = 0; end

PcSize = 0.35;
N = 20;
addpath('functions');

fdir = fullfile('..','analysis', 'derdikman');
% mkdir(fdir);

fmain = fullfile(fdir, sprintf('model_PcSize%0.2f_seed%03d.mat', PcSize, seed));
if exist(fmain, 'file')
    models = load(fmain);
    [turning1, alternating1] = get_arms(1);
    [turning2, alternating2] = get_arms(2);
    return;
end

fname = fullfile('..','analysis','base', sprintf('Size%0.2f_Nx%d_Ny%d', PcSize, N, N), sprintf('seed%03d.mat', seed));
mkdir(fullfile('..','analysis','base', sprintf('Size%0.2f_Nx%d_Ny%d', PcSize, N, N)));
config = struct('fname', fname, 'rng_seed', seed, 'PcSize', PcSize, 'paint', 0, 'Nx', N, 'Ny', N);
gen_base(config);

f = load(fname);
J = f.J;
Cv = f.Cov;

cellPositions_X = 1:20;
cellPositions_Y = 1:20;
[X, Y] =ndgrid(cellPositions_X,cellPositions_Y); %need for plotting later the grid
xy = [reshape(X,1,[]); reshape(Y,1,[])];
[P0, lxy] = get_xy2P(xy);
cost = .3;
D0 = Cv;

[turning1, alternating1] = get_arms(1);
[model] = create_hairpin(1, P0, lxy, cost, D0, J, 0);
models.toWest = model;


[turning2, alternating2] = get_arms(2);
[model] = create_hairpin(2, P0, lxy, cost, D0, J, 1);
models.toEast = model;

models.base = struct('J', f.J, 'Gridness60', f.Gridness60, 'Cov', f.Cov);
save(fmain, '-struct', 'models');
end

function [model] = create_hairpin(mode, P0, lxy, cost, D0, J, do_plot)

xy = lxy(:, 2:3)';
L0 = exp(cost)*eye(size(P0)) - P0;

n = round(sqrt(size(P0, 1)));
[start, acts, wall] = hairpin_define(n);
if mode==1
    goal = [1 1];
else
    goal = [1 n];
end
s = find(sum((xy' - start).^2, 2)==0);
g = find(sum((xy' - goal).^2, 2)==0);

actions = [];
walls = [];
for i=1:length(acts)
    actions = [actions acts{i}];
    walls = [walls wall{i}];
end
[P_all] = create_object(P0, lxy, s, actions, walls);
cexp = cost*ones(size(P_all, 1), 1);
L = diag(exp(cexp)) - P_all;
R_all = L - L0;
C_all = eye(size(P_all));
    
if do_plot
    nr = 2;
    nc = 5;
    fsiz = [0 0 1 1];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
end

if mode == 2
    a{1} = acts{2};
    a{2} = [acts{3} acts{4}];
    a{3} = [acts{5} acts{6}];
    a{4} = [acts{7} acts{8}];
    a{5} = [acts{9} acts{10}];
    
    w{1} = wall{2};
    w{2} = [wall{3} wall{4}];
    w{3} = [wall{5} wall{6}];
    w{4} = [wall{7} wall{8}];
    w{5} = [wall{9} wall{10}];
    
    [~, trajectory] = create_object(P0, lxy, s, acts{1}, wall{1});
    s = trajectory(end);

else
    a{1} = [acts{2} acts{3}];
    a{2} = [acts{4} acts{5}];
    a{3} = [acts{6} acts{7}];
    a{4} = [acts{8} acts{9}];
    a{5} = acts{10};
    
    w{1} = [wall{2} wall{3}];
    w{2} = [wall{4} wall{5}];
    w{3} = [wall{6} wall{7}];
    w{4} = [wall{8} wall{9}];
    w{5} = wall{10};
    
    [~, trajectory] = create_object(P0, lxy, s, acts{1}, wall{1});
    s = trajectory(end);
end


Y = [];
A = [];
rr = [];
cc = [];
all_barrier_index = [];
for i=1:length(a)
    [P, trajectory] = create_object(P0, lxy, s, a{i}, w{i});
    s = trajectory(end);

    if do_plot
        subplot(nr ,nc, i);
        plot_walls(P, lxy, []);    
    end

    R = P0 - P;
    C = eye(size(P));
    
    barrier_index = sum(abs(R), 2)>0;
    r_i = R(barrier_index, :);
    c_i = C(:, barrier_index);

    all_barrier_index = cat(2, all_barrier_index, find(barrier_index)');
    
    Y_i = -r_i*D0*c_i;
    A_i = (eye(size(Y_i))-Y_i)^-1;    
    A = blkdiag(A, A_i);    

end

r = R_all(all_barrier_index, :);
c = C_all(:, all_barrier_index);

Y = -r*D0*c;
A = (eye(size(Y)) - Y)^-1;

U = J - D0 * c * A * r * J;
model = struct('lxy', lxy, 'P', P_all, 'cost', cost, 'D0', D0,'A', A, 'c', c, 'r', r, 'U', U, 'J', J, 'learning', 'initial');

end

function [start, acts, wall] = hairpin_define(n)
start = [1, 1];
num_arms = 10;
sq = round(n/num_arms);

acts0 = [repmat('u',1,n-1), '0', repmat('r',1,n-1), '0', repmat('d',1,n-1), '0'];
wall0 = [repmat('l',1,n-1), 'l', repmat('u',1,n-1), 'u', repmat('r',1,n-1), 'r'];

acts0 = [acts0, repmat('l',1,n-1), '0', repmat('r',1,n-sq)];
wall0 = [wall0, repmat('d',1,n-1), 'd', repmat('0',1,n-sq)];

acts = {acts0}; 
wall = {wall0};

for i = 1:4  
    acts{end+1} = [repmat('u',1,n-sq-1) '0' repmat('u',1,sq) repmat('l',1,sq)];
    wall{end+1} = [repmat('l',1,n-sq-1) 'l' repmat('0',1,sq) repmat('0',1,sq)];
    acts{end+1} = [repmat('d',1,n-sq-1) '0' repmat('d',1,sq) repmat('l',1,sq)];
    wall{end+1} = [repmat('l',1,n-sq-1) 'l' repmat('0',1,sq) repmat('0',1,sq)];
end

acts{end+1} = [repmat('u',1,n-sq-1) '0' repmat('u',1,sq) repmat('l',1,sq)];
wall{end+1} = [repmat('l',1,n-sq-1) 'l' repmat('0',1,sq) repmat('0',1,sq)];

end

function [turning, alternating] = get_arms(mode)

if mode == 2
v1 = [26:39 59:-1:46 ];
turning1 = {v1, v1+80, v1+160 v1+240 v1+320};
v2 = [75:-1:62 82:95];
turning2 = {v2 , v2+80, v2+160 v2+240};
turning = [turning1, turning2];
end

if mode == 1
v1 = [95:-1:82 62:75];
turning1 = {v1, v1+80, v1+160 v1+240};
v2 = [46:59 39:-1:26];
turning2 = {v2+320, v2+240, v2+160, v2+80, v2};
turning = [turning1, turning2];
end

v = [43:58 63:78];
% v = [41:60; 61:80];
if mode == 2
alternating_even = {v, v+80, v+160, v+240, v+320};
alternating_odd = {v-40, v+40, v+120, v+200, v+280};
end

if mode == 1
alternating_odd = {v+320, v+240, v+160, v+80, v};
alternating_even = {v+280, v+200, v+120, v+40, v-40};
end
alternating = [alternating_odd', alternating_even'];
alternating = reshape(alternating', 1, numel(alternating));
end

function B = learning(A_init, r, c, D0, alf, num_steps)
Y = -r*D0*c;
I = eye(size(Y));

A = (I - Y)^-1;

B = A_init;
for i=1: num_steps
    B = B + alf*(I + B*Y - B);
    e(i) = max(max(abs(A-B)));
end


end

function [P, trajectory] = create_object(P0, lxy, start, actions, wall)

A = double(P0 > 0); 

N = size(lxy,1);
s = start;
len_barrier = numel(actions);

wall_down = false(N,1); 
wall_up = false(N,1);
wall_right = false(N,1);
wall_left = false(N,1);

trajectory = s;

for k = 1:len_barrier
    
  xys = lxy(s, 2:end);
  a = actions(k);
  
  if strcmp(a,'u')
    dxy = [1 0];
  elseif strcmp(a,'r') 
    dxy = [0 1];
  elseif strcmp(a,'d')
    dxy = [-1 0];
  elseif strcmp(a,'l')
    dxy = [0 -1];
  elseif strcmp(a,'0')
    dxy = [0 0];
  else
    error('Unknown action')
  end

  w = wall(k);

  if strcmp(w,'u')
    dw = [1 0];
  elseif strcmp(w,'d')
    dw = [-1 0];
  elseif strcmp(w,'r')
    dw = [0 1];
  elseif strcmp(w,'l')
    dw = [0 -1];
  elseif strcmp(w,'0')
    dw = [];
  else
    error('Unknown wall!')
  end

  s_old = s;
  xy_next = xys + dxy;
  
  s = find(all(lxy(:,2:end) == xy_next,2));
  if isempty(s)
    s = 1;
  else 
    s = s(1);
  end
  
  trajectory = [trajectory; s];
  
  if ~isempty(dw)
    xy_discont = xys + dw;
    s_discont = find(all(lxy(:,2:end) == xy_discont,2));
    A(s_old, s_discont) = 0;
    A(s_discont, s_old) = 0;
  end

  wall_down(s) = logical(w=='d');
  wall_up(s) = logical(w=='u');
  wall_right(s) = logical(w=='r');
  wall_left(s) = logical(w=='l');
  
end

deg = sum(A,2);
P = diag(1./deg) * A;

end
