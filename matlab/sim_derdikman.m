function sim_derdikman(draw_main, draw_supplementary)
if nargin<2, draw_main = 1; draw_supplementary = 0; end

[examplar, turning_mat, diag_turning_mat, corr_mat] = run_stats();
P = examplar.P;
lxy = examplar.lxy;
activity1 = examplar.activity1;
activity2 = examplar.activity2;
mean_turning_diag = mean(diag_turning_mat, 2);
std_turning_diag = serr(diag_turning_mat, 2);

if draw_supplementary
    fsiz = [0 0 .5 .4];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);    
    
    plot_activity_all(activity1, P, lxy, 5, 10)
end

if draw_main

    fsy = 16;
    
    
    fsiz = [0 0 .29 .29];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    imagesc(turning_mat, [-.2 1]);
    colormap('jet')
    hc = colorbar;
    axis equal;
    axis image;
    set(gca, 'xtick', [], 'ytick', [], 'xgrid', 'on', 'ygrid', 'on')
    set(gca,'fontsize', fsy)
    pos = get(hc, 'position');
    pos(1) = pos(1)+.04;
    pos(3) = pos(3)*1.3;
    % pos(2) = pos(2) + 0.02;
    % pos(4) = pos(4)*.98;
    set(hc, 'position', pos);
    set(gca, 'linewidth', 1.5, 'TickLength', [0 0]);
    
    
    xlim = get(gca, 'xlim');
    ylim = get(gca, 'ylim');
    hold on;
    plot(xlim, ylim, 'color', 'white', 'linewidth', 1)
    hold on;
    plot(xlim+[2 0], ylim+[0 -2], 'color', 'white', 'linewidth', 1)
    
    plot(xlim, [0, 0]+mean(xlim), 'color', 'white', 'linewidth', 1)
    plot([0, 0]+mean(ylim), ylim, 'color', 'white', 'linewidth', 1)
    
    xl = (1:size(turning_mat, 2)) - floor(size(turning_mat, 2)/2);
    fsiz = [0 0 .2 .22];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    
    plot(xl(1:end-1), mean_turning_diag, 'color', 'k', 'linewidth', 2);
    errorbar(xl(1:end-1), mean_turning_diag, std_turning_diag, 'color', 'k', 'linewidth', 2);
    set(gca,"FontSize", fsy, 'box', 'off', 'linewidth', 1, 'ylim', [0 1]);
    % xlabel('Distance from turnpoint', 'FontSize', fsy);
    % ylabel('Diagonal correlation value', 'FontSize', fsy);
    % 
    % 
    fsiz = [0 0 .31 .31];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    imagesc(corr_mat, [-1 1]);
    colormap('jet')
    hc = colorbar;
    axis equal;
    axis image;
    set(gca,'fontsize', fsy)
    pos = get(hc, 'position');
    pos(1) = pos(1)+.04;
    pos(3) = pos(3)*1.3;
    pos(2) = pos(2) + 0.02;
    pos(4) = pos(4)*.98;
    set(hc, 'position', pos);
    xlabel('Arm number', 'fontsize', fsy)
    ylabel('Arm number', 'fontsize', fsy)
    set(gca, 'linewidth', 1.5, 'TickLength', [0 0]);
    
    idx1 = [6 14]; %29
    idx2 = [12 19]; %39, 40
    
    Z = [activity1(:, idx1) activity2(:, idx2)];
    
    fsiz = [0 0 .2 .35];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_activity(Z, P, lxy, 2, 2);
    
end

end

function [exemplar, turning_mat, diag_turning_mat, median_corrmat] = run_stats()
fname = fullfile('sum', sprintf('derdikman.mat'));

save_it = ~exist(fname, 'file');
if save_it
    [models, turning1, alternating1, turning2, alternating2] = derdikman_make();
    
    U1 = models.toWest.U;    
    U2 = models.toEast.U;
    
    P = models.toWest.P;
    lxy = models.toWest.lxy;
    
    [~, activity2] = derdikman_measures(U2, turning2, alternating2, 1);
    [~, activity1] = derdikman_measures(U1, turning1, alternating1, 1);
    
    exemplar = struct('P', P, 'lxy', lxy, 'activity1', activity1, 'activity2', activity2);

    alf = 1;
    num_sim = 50;    
    for i=1:num_sim
    
        [models, turning1, alternating1, turning2, alternating2] = derdikman_make(i);   
    
        U1 = models.toWest.U;    
        U2 = models.toEast.U;
            
        [corr_mat2, activity2, turning_mat2] = derdikman_measures(U2, turning2, alternating2, alf);
        [corr_mat1, activity1, turning_mat1] = derdikman_measures(U1, turning1, alternating1, alf);
    
        tm = cat(3, turning_mat1, turning_mat2);
        turning_mat(:, :, i) = mean(tm, 3);
        diag_turning_mat(:, i) = diag(mean(tm, 3), 1);
    
        num_alternating_PCs(i, :) = [size(activity1, 2), size(activity2, 2)];
        
        C = cat(3, corr_mat1, corr_mat2);
        median_corrmat(:, :, i) = mean(C, 3);    
    end

    save(fname, 'exemplar', 'turning_mat' , 'median_corrmat', 'num_alternating_PCs', 'diag_turning_mat');
end

f = load(fname);

turning_mat = f.turning_mat;
diag_turning_mat = f.diag_turning_mat;
% gridness = f.gridness;

median_corrmat = f.median_corrmat;
median_corrmat = mean(median_corrmat, 3);
turning_mat = mean(turning_mat, 3);
exemplar = f.exemplar;
end

% -------------------------------------------------------------------------
function plot_activity(U, P, lxy, plt_nr, plt_nc)

n = round(sqrt(size(P, 1)));
for i= 1:size(U, 2)
    subplot(plt_nr,plt_nc,i);

    C = zeros(n,n);
    C(:) = U(:,i);
    
    imagesc(C);
    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
%     colormap('jet');
    axis image;
%     colorbar

    hold on;
    config = struct('linewidth', 2, 'linecolor', 'w');    
    plot_walls(P, lxy, config);

%     scatter(x, y, 20, 'MarkerFaceColo', 'r', 'MarkerFaceAlpha', .4);

end

end

function plot_activity_all(U, P, lxy, plt_nr, plt_nc)
K = min(size(U, 2), plt_nr*plt_nc);
n = round(sqrt(size(U, 1)));
for i= 1:K
    subplot(plt_nr,plt_nc,i);

    C = zeros(n,n);
    C(:) = U(:,i);
    
    imagesc(C);
    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
    colormap('jet');
    axis image;
%     colorbar

    if ~isempty(P)
        hold on;
        plot_walls(P, lxy);
    end
%     scatter(x, y, 20, 'MarkerFaceColo', 'r', 'MarkerFaceAlpha', .4);

end

end
