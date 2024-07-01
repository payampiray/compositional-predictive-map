function [P, lxy] = get_xy2P(xy)
N = size(xy, 2);
P = zeros(N, N);
eps = .01;
for i=1:N
    dist = sum((xy(:, i) - xy).^2, 1);
    dist(i) = 100;
    sucs = abs(dist - min(dist))< eps;
    P(i, sucs) = 1;
end

deg = sum(P>0, 2);
P = diag(deg.^-1)*P;

lxy = [(1:N)' xy'];

end
