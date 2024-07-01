function [task, model] = brecht_make(home, seed)
if nargin==0
    home = 2;    
end
if nargin<2, seed = 0; end

PcSize = .35;

cost = .1;
Nx = 24;
Ny = 24;
fname = fullfile('..','analysis', 'base', sprintf('Size%0.2f_Nx%d_Ny%d', PcSize, Nx, Ny), sprintf('seed%03d.mat', seed));
conf = struct('fname', fname, 'rng_seed', seed, 'PcSize', PcSize, 'paint', 0, 'Nx', Nx, 'Ny', Ny, 'step_cost', cost);
f = gen_base(conf);

cellPositions_X = 1:Nx;
cellPositions_Y = 1:Ny;
[X, Y] =ndgrid(cellPositions_X,cellPositions_Y); %need for plotting later the grid
xy = [reshape(X,1,[]); reshape(Y,1,[])];
[P0] = get_xy2P(xy);

if home == 1
    idx1 = find( (xy(1, :) > 7).*(xy(1, :)<=11).*xy(2,:)==19);
    idx2 = find( (xy(1, :) > 7).*(xy(1, :)<=11).*xy(2,:)==20);

    idx1 = [idx1 find( (xy(1, :) > 13).*(xy(1, :)<=17).*xy(2,:)==19)];
    idx2 = [idx2 find( (xy(1, :) > 13).*(xy(1, :)<=17).*xy(2,:)==20)];
% 
    idx1 = [idx1 find( (xy(1, :) == 7).*(xy(2, :)>=20).*(xy(2, :)<25))];
    idx2 = [idx2 find( (xy(1, :) == 8).*(xy(2, :)>=20).*(xy(2, :)<25))];
%     
    idx1 = [idx1 find( (xy(1, :) == 17).*(xy(2, :)>=20).*(xy(2, :)<25))];
    idx2 = [idx2 find( (xy(1, :) == 18).*(xy(2, :)>=20).*(xy(2, :)<25))];

    idx_center = find( (xy(1, :) > 10).*(xy(1, :)<15).*(xy(2,:)==23) ) ;

end

if home == 2
    idx1 = find( (xy(1, :) > 10).*(xy(1, :)<=14).*xy(2,:)==16);
    idx2 = find( (xy(1, :) > 10).*(xy(1, :)<=14).*xy(2,:)==17);
    
    idx1 = [idx1 find( (xy(2, :) > 8).*(xy(2, :)<=11).*xy(1,:)==10)];
    idx2 = [idx2 find( (xy(2, :) > 8).*(xy(2, :)<=11).*xy(1,:)==11)];

    idx1 = [idx1 find( (xy(2, :) > 13).*(xy(2, :)<=16).*xy(1,:)==10)];
    idx2 = [idx2 find( (xy(2, :) > 13).*(xy(2, :)<=16).*xy(1,:)==11)];

    idx1 = [idx1 find( (xy(2, :) > 8).*(xy(2, :)<=11).*xy(1,:)==14)];
    idx2 = [idx2 find( (xy(2, :) > 8).*(xy(2, :)<=11).*xy(1,:)==15)];    

    idx1 = [idx1 find( (xy(2, :) > 13).*(xy(2, :)<=16).*xy(1,:)==14)];
    idx2 = [idx2 find( (xy(2, :) > 13).*(xy(2, :)<=16).*xy(1,:)==15)];    

    idx1 = [idx1 find( (xy(1, :) > 10).*(xy(1, :)<=14).*xy(2,:)==9)];
    idx2 = [idx2 find( (xy(1, :) > 10).*(xy(1, :)<=14).*xy(2,:)==8)];

    idx_center = find( (xy(1, :) > 11).*(xy(1, :)<14).*(xy(2,:)>11).*(xy(2,:)<=13) ) ;
end

idx_edge = [find( (xy(1, :) > 0).*(xy(1, :)<25).*(xy(2,:)==1) ) ];
idx_edge = [idx_edge find( (xy(1, :) > 0).*(xy(1, :)<25).*(xy(2,:)==24) ) ];
idx_edge = [idx_edge find( (xy(2, :) > 0).*(xy(1, :)<25).*(xy(2,:)==1) ) ];
idx_edge = [idx_edge find( (xy(2, :) > 0).*(xy(1, :)<25).*(xy(2,:)==24) ) ];

P = P0;
for i=1:length(idx1)
    P(idx1(i), idx2(i)) = 0;
    P(idx2(i), idx1(i)) = 0;
end

P = P>0;
deg = sum(P, 2);
P = diag(deg.^-1)*P;

N = size(P, 2);
lxy = [(1:N)' xy(2,:)' xy(1,:)'];

P_plain_box = P;
for i=idx_center
    P(i, :) = 0;
    P(i, i) = 1;

end

D0 = f.Cov;
J = f.J;

U = calculate_U(D0, P0, J, P);
U_plain = calculate_U(D0, P0, J, P_plain_box);

task = struct('home', home, 'xy', xy, 'P', P, 'P_plain_box', P_plain_box,'lxy', lxy, 'J', J, 'D0', D0, 'idx_edge', idx_edge);
model = struct('U', U, 'U_plain', U_plain);

end

function U = calculate_U(D0, P0, J, P)
R = P0 - P;
C1 = eye(size(P));

barrier_index = sum(abs(R), 2)>0;
r = R(barrier_index, :);
c = C1(:, barrier_index);

Y = -r*D0*c;
A = (eye(size(Y))-Y)^-1;
U = J - D0 * c * A * r * J;
end
