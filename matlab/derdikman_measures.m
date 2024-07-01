function [corr_all, eigvec_all, turning_mat] = derdikman_measures(eigvec_all, turning, alternating, alf)
if nargin<4, alf = 0.05; end

num_cols = 10; 
K = size(eigvec_all,2);
N = size(eigvec_all,1);
n = floor(sqrt(N));

corr_all = zeros(num_cols,num_cols,K);
activity_all = zeros(n,n,K);
corr_p = zeros(K,1);

for j = 1:K
    x = eigvec_all(:,j);
    activity = zeros(n,n);
    for i = 1:n
        k = (i-1)*n + 1:i*n;
        activity(:,i) = x(k); 
    end
    activity_all(:,:,j) = activity;    
%     C = corr(mask(activity));
%     [corr_p(j), ma(j), mb(j)] = diag_stats(C);
%     corr_all(:,:,j) = C;

    for l=1:length(alternating)
        a = activity(alternating{l});
        masked(:, l) = a;
    end
    C = corr(masked);
    [corr_p(j), ma(j), mb(j)] = diag_stats(C);
    corr_all(:,:,j) = C;    
end

idx = find(corr_p < .025);
ma = ma(idx);
mb = mb(idx);
mab = [ma; mb]';

U = eigvec_all;
corr_all = corr_all(:,:,idx); 
% activity_all = activity_all(:,:,idx);
% eigvec_all = eigvec_all(:,idx);

% U(:, idx) = [];

Um = U;
Um(:, idx) = [];
Um = mean(Um, 2);

num_arms = length(turning);
for i= 1:num_arms
    turn = turning{i};
    middle = length(turn)/2 + (-1:2);
    u_middle = median(U(turn(middle), :), 1);
    i_turn1 = 1:(middle(1)-1);
    i_turn2 = (middle(end)+1):length(turn);
    u = [U(turn(i_turn1), :); u_middle; U(turn(i_turn2), :)];    
    all_turning_activity(:, :, i) = u;
    
    u_middle = median(Um(turn(middle), :), 1);
    u = [Um(turn(i_turn1), :); u_middle; Um(turn(i_turn2), :)];    
    turning_activity(:, :, i) = u;

%     u = Um(turn, :);
%     u(middle, :) = [];
%     turning_activity(:, :, i) = u;

end

siz = size(turning_activity);
v = nan(siz([1 3]));
v(:) = median(turning_activity(:, 1, :), 2);    
turning_mat = corr(v', v');

% siz = size(all_turning_activity);
% for i= 1:siz(2)
%     v = nan(siz([1 3]));
%     v(:) = median(all_turning_activity(:, i, :), 2);    
%     turning_mat(:, :, i) = corr(v', v');
% end
% turning_mat = mean(turning_mat, 3);

% siz = size(all_turning_activity);
% v = nan(siz(1), siz(2)*siz(3));
% v(:) = reshape(all_turning_activity, siz(1), siz(2)*siz(3));
% turning_mat = corr(v', v');

% figure;
% imagesc(turning_mat, [-1 1]);
% colorbar;
% axis image;


end

function [p, ma, mb] = diag_stats(corr_mat)

aa = []; 
bb = [];
idx_a = [2 4 6 8]; 
idx_b = [1 3 5 7];
for i = 1:numel(idx_a)
   a = diag(corr_mat,idx_a(i)); 
   aa = [aa; a];
end
for i = 1:numel(idx_b)
   b = diag(corr_mat,idx_b(i));
   bb = [bb; b]; 
end
p = ranksum(aa,bb,'tail','right');
ma = median(aa);
mb = median(bb);
end