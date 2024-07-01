function f = boccara_make(seed)
if nargin<1, seed = 0; end

PcSize = 0.35;
addpath('functions');


fmodel = fullfile('..','analysis', 'boccara', sprintf('model_seed%03d_Size%0.2f_N25.mat', seed, PcSize));   
if exist(fmodel, 'file')
    f = load(fmodel);
    return;
end

fdir = fullfile('..','analysis', 'base', sprintf('Size%0.2f_Nx25_Ny25', PcSize));
fname = fullfile(fdir, sprintf('seed%03d.mat', seed));
config = struct('fname', fname, 'rng_seed', seed, 'PcSize', PcSize, 'paint', 0);
gen_base(config);

f = load(fname);
f = rmfield(f, {'outputMap', 'autocorrMap'});
xy = f.placeCenters;
Cv = f.Cov;


[task, model] = create_Bocarra(xy, Cv);
f.task = task;    

J = f.J;
U = J - model.D0 * model.c * model.A * model.r * J;
model.U = U;

[~, ~, model.Gridness60, model.Gridness90] = Maps(f, U);        
f.model = model;    

save(fmodel, '-struct', 'f');
end

function [config, model] = create_Bocarra(xy, Cv)
thr_goal = .25;

cost = .3;
[P0, lxy] = get_xy2P(xy);
D0 = Cv;

P = P0;

goals = {[7 4], [6 7], [3 8]};
goals_index = [];
for k=1:3
    goal = goals{k};    
    dist = sum((lxy(:, 2:end) - goal).^2, 2);
    [sorted_dist, index_sorted_dist] = sort(dist);
    i_goal = index_sorted_dist(sorted_dist< thr_goal);
    goals_index = cat(1, goals_index, i_goal);
end

q = cost*ones(size(P, 1), 1);
q(goals_index) = -1;
P = P0;
P(goals_index, :) = 0;
idx = sub2ind(size(P),goals_index, goals_index);
P(idx) = 1;

L0 = exp(cost)*eye(size(P0)) - P0;

L = diag(exp(q)) - P;
R = L - L0;
barrier_index = find(sum(abs(R), 2)>0);
C = eye(size(P));
r = R(barrier_index, :);
c = C(:, barrier_index);
Y = -r*D0*c;
A = (eye(size(Y))-Y)^-1;    

model = struct('D0', D0, 'A', A, 'r', r, 'c', c);
config = struct('goals', {goals}, 'P0', P0, 'lxy', lxy, 'cost', cost, 'P', P, ...
                'thr_goal', thr_goal, 'goals_center', {goals}, 'goals_index', goals_index);


end
