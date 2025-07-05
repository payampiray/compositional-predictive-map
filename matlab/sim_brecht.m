function sim_brecht(draw_main, draw_supplementary)
if nargin<2, draw_main = 1; draw_supplementary = 1; end



[examplar_home1, ~, C_home1, C_plain1, C_home1_plain] = run_stats(1);
[examplar_home2, C_home2_all, C1_home2, ~, ~, rate1_home2, rate0_home2] = run_stats(2);
rate1 = rate1_home2;
rate0 = rate0_home2;

U_home2 = examplar_home2.model.U;
J_home2 = examplar_home2.task.J;
P_home2 = examplar_home2.task.P_plain_box;
lxy_home2 = examplar_home2.task.lxy;
peakrate1_home2 = examplar_home2.peakrate1;
peakrate0_home2 = examplar_home2.peakrate0;


U_home1 = examplar_home1.model.U;
J_home1 = examplar_home1.task.J;
P_home1 = examplar_home1.task.P_plain_box;
lxy_home1 = examplar_home1.task.lxy;


% normalize
U_home1 = U_home1./max(U_home1);
J_home1 = J_home1./max(J_home1);
U_home2 = U_home2./max(U_home2);
J_home2 = J_home2./max(J_home2);

mC_home1 = nanmean(C_home1, 2); %#ok<NANMEAN> 
mC_home2 = nanmean(C1_home2, 2); %#ok<NANMEAN> 

mC_plain1 = nanmean(C_plain1, 2); %#ok<NANMEAN> 
mC_home_plain1 = nanmean(C_home1_plain, 2); %#ok<NANMEAN> 

rate1_mean = mean(rate1);
rate0_mean = mean(rate0);

rate1_serr = serr(rate1);
rate0_serr = serr(rate0);    

% --------------------

if draw_supplementary
    fsiz = [0 0 .7 .4];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);    

    plot_C(C_home2_all);
end

if draw_main 
    
    fs = 16;
    fsy = 24;

    % plot cell activity
    idx_home1 = 4;
    idx_home2 = 10;
    
    activity0 = J_home1(:, idx_home1);
    activity1 = U_home1(:, idx_home1);
    fsiz = [0 0 .07 .18];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_activity(activity0, P_home1, lxy_home1, 0);
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_activity(activity1, P_home1, lxy_home1, 1);
    
    activity0 = J_home2(:, idx_home2);
    activity1 = U_home2(:, idx_home2);
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_activity(activity0, P_home2, lxy_home2, 0);
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_activity(activity1, P_home2, lxy_home2, 1);
    
    
    % plot corr map
    fsiz = [0 0 .2 .22];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_corr_map(mC_home1, P_home1, lxy_home1);
    
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_corr_map(mC_home2, P_home2, lxy_home2);
    
    % plot norm rate as a function of distance
    fsiz = [0 0 .15 .22];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    
    shadedErrorBar(0:6, rate1_mean, rate1_serr, 'lineProps', 'g'); hold on;
    shadedErrorBar(0:6, rate0_mean, rate0_serr, 'lineProps', 'b');
    set(gca, 'fontsize', fs);
    ylabel('Norm rate', 'fontsize', fsy);
    xlabel('Distance to home', 'fontsize', fsy);
    ylim([.8 2.2]);
    
    % plot rate map and their difference
    fsiz = [0 0 .12 .13];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_rate_map(peakrate0_home2, [0 1]);
    
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_rate_map(peakrate1_home2, [0 1], P_home2, lxy_home2);
    
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_rate_map(peakrate1_home2 - peakrate0_home2, [-.4 .3], P_home2, lxy_home2);
    
    % plot corr map plain box
    fsiz = [0 0 .15 .22];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_corr_map(mC_plain1, P_home1, lxy_home1);
    
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_corr_map(mC_home_plain1, P_home1, lxy_home1);

end

end

% -------------------------------------------------------------------------
function [exemplar, C_home_all, C_home, C_plain, C_home_plain, rate1, rate0] = run_stats(home)

fname = fullfile('sum', sprintf('brecht_home%d.mat', home));

num_sim = 50;
save_it = ~exist(fname, 'file');
if save_it
    [exemplar.task, exemplar.model] = brecht_make(home);
    [~, ~, ~, ~, ~, exemplar.peakrate1, exemplar.peakrate0] = brecht_measures(exemplar.task, exemplar.model);

    for i=1:num_sim
        [task, model] = brecht_make(home, PcSize, i);
        [C_home(:, i), C_plain(:, i), C_home_plain(:, i), rate1(:, :, i), rate0(:, :, i), peak_rate1(:, i), peak_rate0(:, i)] = brecht_measures(task, model);
    end

    save(fname, 'exemplar', 'C_home', 'C_plain', 'C_home_plain','rate1', 'rate0', 'peak_rate1', 'peak_rate0');
end

f = load(fname);

C_home_all = f.C_home;
C_plain = f.C_plain;
C_home_plain = f.C_home_plain;
rate1 = f.rate1;
rate0 = f.rate0;

rate1 = mean(rate1, 1);
rate0 = mean(rate0, 1);

num_points = 7;
r1 = zeros(num_points, num_sim);
r1(:) = rate1(1, :, :);
rate1 = r1';

r0 = zeros(num_points, num_sim);
r0(:) = rate0(1, :, :);
rate0 = r0';

C_home = mean(C_home_all, 2);
C_plain = mean(C_plain, 2);
C_home_plain = mean(C_home_plain, 2);

exemplar = f.exemplar;
end

% -------------------------------------------------------------------------
function plot_C(C)

plt_nr = 5;
plt_nc = 10;
n = 24;

K = size(C, 2);
for i= 1:K
    h(i) = subplot(plt_nr,plt_nc,i);

    corr_map = zeros(n,n);
    corr_map(:) = C(:,i);
    imagesc(corr_map', [0 1]);

    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');      
    colormap('jet');
    axis image;
    colorbar(gca);
end


end

function plot_activity(activity, P, lxy, show_home)

U = activity;

ulim = [0 1];

n = sqrt(size(U,1));
for i= 1    

    C = zeros(n,n);
    C(:) = U(:,i);
    imagesc(C', ulim);
    
%     imagesc(xy(1,:), xy(2,:), C);
    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
%     axis equal;
    axis image;
    colormap('jet');

    hc = colorbar;
    set(hc,'location','southoutside','fontsize', 14);
    hc.Label.String = 'Norm. Rate';
    hc.Label.FontSize = 18;

        
    if show_home
        hold on;
        config = struct('linewidth', 4, 'linecolor', 'w');
        plot_walls(P, lxy, config);           
    end


end

end

function plot_corr_map(C, P, lxy)

n = sqrt(size(C,1));

corr_map = zeros(n,n);
corr_map(:) = C;
imagesc(corr_map', [0 1]);   

set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');      
colormap('jet');
axis image;
hc = colorbar;

set(hc,'location','southoutside','fontsize', 14);
hc.Label.String = 'Correlation';
hc.Label.FontSize = 18;

hold on;
config = struct('linewidth', 4, 'linecolor', 'k');
plot_walls(P, lxy, config);   

end


function plot_rate_map(peak_rate, ulim, P, lxy)

n = 24;

C = zeros(n,n);
C(:) = peak_rate;

imagesc(C', ulim);

set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
axis equal;
axis image;
colormap('jet');

if nargin>2
    hold on;
    config = struct('linewidth', 4, 'linecolor', 'k');
    plot_walls(P, lxy, config);            
end

end