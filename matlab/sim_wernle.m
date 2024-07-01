function sim_wernle(draw_main, draw_supplementary)
if nargin<2, draw_main = 1; draw_supplementary = 0; end

[exemplar, mC] = run_stats();
task = exemplar.task;
model = exemplar.model;
C = exemplar.C;

lxy = task.lxy;
P0 = task.P0;
U = model.U;
J = task.J;

if draw_main
    idx = [4 19]; %16    
    fsiz = [0 0 .7 .4];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_maps(J(:, idx), U(:, idx), C(:, idx), mC, P0, lxy);
end

if draw_supplementary
    fsiz = [0 0 .7 .4];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    
    plot_corr_maps(C);
end

end

% -------------------------------------------------------------------------
function [exemplar, mC] = run_stats()
num_sim = 50;

fname = fullfile('sum', 'wernle.mat');

save_it = ~exist(fname, 'file');
if save_it
    [exemplar.task, exemplar.model] = wernle_make();
    [exemplar.C] = wernle_measures(exemplar.task, exemplar.model);
    for i=1:num_sim
        [task, model] = wernle_make(i);
        [corr_map(:, :, i)] = wernle_measures(task, model);
    end
    save(fname, 'exemplar', 'corr_map');
end

f = load(fname);

corr_map = f.corr_map;
mC = nanmedian(corr_map, 2);
mC = mean(mC, 3);

exemplar = f.exemplar;
end
% -------------------------------------------------------------------------
function [corr_map] = wernle_measures(task, model)
box_size = 6;

lxy = task.lxy;
J = task.J;
U = model.U;

xy = lxy(:, 2:3)';
for i=1:size(U, 1)
    dist = xy - xy(:, i);
    sucessors = abs(dist)<=box_size;
    sucessors = sum(sucessors, 1)==2;
    
    r = corr(U(sucessors, :), J(sucessors, :), 'type', 'Spearman');    
    corr_map(i, :) = diag(r);
end

end

% -------------------------------------------------------------------------

function plot_maps(J, U, C, mC, P, lxy)

plt_nr = 2;
plt_nc = 4;
n = sqrt(size(U, 1));

Z = [J(:, 1) U(:, 1) C(:, 1) mC J(:, 2) U(:, 2) C(:, 2) ];
K = size(Z, 2);

sub_plots = [1 2 3 4 5 6 7];

fs = 14;

for i= 1:K
    subplot(plt_nr,plt_nc, sub_plots(i));
%     figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);


    map = zeros(n,n);
    map(:) = Z(:,i);
    if i == 3 || i == 7        
        imagesc(map', [-.3 1]);
    elseif i == 4
        imagesc(map', [.4 .8]);
    else
        imagesc(map');
    end

    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
    colormap('jet');
    axis image;
    if any(i == [3 7])
        hc = colorbar;
        set(hc, 'Ticks', [-.3 1], 'Fontsize', fs);
    end
    if i == 4
        hc = colorbar;
        set(hc, 'Ticks', [.4 .8], 'Fontsize', fs);
    end

    if i==1 || i==5
        hold on;
        config = struct('linewidth', 3, 'linecolor', 'w');
        plot_walls(P, lxy, config);                   
    end

end


end

% -------------------------------------------------------------------------
function plot_corr_maps(C)

plt_nr = 5;
plt_nc = 10;
n  = sqrt(size(C, 1));

K = size(C, 2);
for i= 1:K
    h(i) = subplot(plt_nr,plt_nc,i);

    corr_map = zeros(n,n);
    corr_map(:) = C(:,i);
    imagesc(corr_map', [-.3 1]);        
    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
    colormap('jet');
    axis image;
    colorbar;
end


end
