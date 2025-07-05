function sim_boccara(draw_main, draw_supplementary)
if nargin<2, draw_main = 1; draw_supplementary = 1; end

[f, increase_activity, med_peak_post, med_peak_pre, correlation, grid_post, grid_pre] = run_stats();

task = f.task;
model = f.model;
U = model.U;
J = f.J;
attraction = f.attraction;
distance = f.distance;

[corr_all, corr_mean] = measure_corr(task, model, J);

% -------------------------------------------------------------------------
ma = mean(increase_activity, 1);
ea = serr(increase_activity, 1);
md = mean([med_peak_post, med_peak_pre], 1);
ed = serr([med_peak_post, med_peak_pre], 1);

mg = mean([grid_pre, grid_post], 1);
eg = serr([grid_pre, grid_post], 1);

mc = mean(correlation, 1);
ec = serr(correlation, 1);

[data_ma, data_md, data_mg, data_eg] = boccara_fig_data();

% grid_score = [gridU(1:48); gridJ(1:48)];
if draw_supplementary
    fsiz = [0 0 .6 .55];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz)
    
    idx = [6, 40 26, 40, 15, 44, 27, 32];
    plot_vectors(J(:, idx), U(:, idx), corr_all(:, idx), task);
    
    fsiz = [0 0 .08 .15];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz)
    plot_corr_maps(corr_mean, task, 1, 1);

end

if draw_main
    
    plot_bars(ma, ea, mg, eg, mc, ec, attraction, distance)
    
    plot_bars(data_ma, 0, data_mg, data_eg)
    
    idx = 6;
    
    fsiz = [0 0 .15 .35];
    figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
    plot_maps(J(:, idx), U(:, idx), task);
end

end

function [r_rank, c_rank] = measure_corr(task, model, J)
box_size = 1.5;

lxy = task.lxy;
U = model.U;

xy = lxy(:, 2:3)';
for i=1:size(U, 1)
    dist = xy - xy(:, i);
    sucessors = abs(dist)<=box_size;
    sucessors = sum(sucessors, 1)==2;
    
    r = corr(U(sucessors, :), J(sucessors, :), 'type', 'Spearman');    
    r_rank(i, :) = diag(r);

    y = U(sucessors, :); y = y(:);
    z = J(sucessors, :); z = z(:);
    c_rank(i, 1) = corr(y, z, 'type', 'Spearman');    
end

end

function [exemplar, increase_activity, peak_post, peak_pre, correlation, grid_post, grid_pre] = run_stats()

fname = fullfile('sum','boccara.mat');
save_it = ~exist(fname, 'file');

if save_it
    f = boccara_make();
    exemplar.task = f.task;
    exemplar.model = f.model;
    exemplar.U = f.model.U;
    exemplar.J = f.J;
    [exemplar.increase_activity, exemplar.med_peak_post, exemplar.med_peak_pre, exemplar.correlation, ...
        exemplar.attraction, exemplar.distance, exemplar.grid_post, exemplar.grid_pre] = boccara_measures(f);

    num_sim = 50;
    increase_activity = nan(num_sim, 1);
    peak_post = nan(num_sim, 1);
    peak_pre = nan(num_sim, 1);
    correlation = nan(num_sim, 1);
    grid_pre = nan(num_sim, 1);
    grid_post = nan(num_sim, 1);
    
    for i= 1:num_sim
        f = boccara_make(i);        
        [increase_activity(i), peak_post(i), peak_pre(i), correlation(i), ~, ~, grid_post(i), grid_pre(i)] = boccara_measures(f);    
    end
    
    save(fname, 'increase_activity', 'peak_post', 'peak_pre', 'correlation', 'grid_pre', 'grid_post');
end

f = load(fname);

increase_activity = f.increase_activity;
peak_post = f.peak_post;
peak_pre = f.peak_pre;
correlation = f.correlation;
grid_post = f.grid_post;
grid_pre = f.grid_pre;
exemplar = f.exemplar;
end

function plot_bars(ma, ea, mg, eg, mc, ec, attraction, distance)


plot_scatter = 1;
if nargin<7
    plot_scatter = 0;
end

fsy = 18;
colmap = [0    0.4470    0.7410; 0.8500    0.3250    0.0980]; 
col0 = [142 118 177]/255;
lw = 1;

nr = 1;
nc = 3;
fsiz = [0 0 .5 .22];
figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);

subplot(nr, nc, 1);
errorbarKxN(ma, ea, {''}, col0)
ylim([0 .9]);
ylabel(sprintf('Proportion of fields\n moving closer to goals'), 'fontsize', fsy);
% set(gca, 'linewidth', lw);

% subplot(nr, nc, 2);
% errorbarKxN(md', ed', {'', ''}, colmap)
% % set(gca,'xticklabel',{'Open field', 'Post-learning'}, 'fontsize', fsy);
% ylabel(sprintf('Peak activity in \nvicinity of goal location'), 'fontsize', fsy);
% % set(gca, 'linewidth', lw);
% 
% % if ~plot_scatter
% % ylim([0 .99]);    
% % legend({'Open field', 'Post-learning'}, 'fontsize', fsy, 'location', 'northwest', 'box', 'off')
% % end

subplot(nr, nc, 2);
errorbarKxN(mg', eg', {'', ''}, colmap);
ylabel('Grid score', 'fontsize', fsy);
% set(gca, 'linewidth', lw);
if ~plot_scatter
ylim([0 .89]);   
legend({'Open field', 'Post-learning'}, 'fontsize', fsy, 'location', 'northwest', 'box', 'off')
end


% % shadedErrorBar(xa, xa, sa, 'lineProps', '-r');

if plot_scatter
subplot(nr, nc, 3);
scatter(distance, attraction, 20, 'k', 'LineWidth',1.5);
hl = lsline; hold on;
set(hl, 'linewidth', 1);
% set(gca, 'linewidth', lw);
scatter(distance, attraction, 20, 'k', 'LineWidth',1.5);
% xlabel('Distance to goal', 'fontsize', fsy);
% ylabel('Attraction strength', 'fontsize', fsy);


pos = get(gca,'position');
pos12 = pos(1:2)+pos(3:4).*[.8 .65];
pos34 = [.15 .3].*pos(3:4);
pos = [pos12 pos34];
axes('Position', pos,'box', 'on');
errorbarKxN(mc, ec, {''},col0, .5)
ylabel('Correlation', 'fontsize', 14);
set(gca, 'box', 'on', 'ylim', [1.5*mc 0 ]);
end

end

function plot_vectors(J, U, corr_map, task)

goals = task.goals_center;
xy = task.lxy(:, 2:3);
n  = round(sqrt(size(xy, 1)));

for i = 1:length(goals)
    goal = goals{i};
    [~, g] = min(sum((xy - goal).^2, 2));
    goal_index = zeros(n*2, 1);
    goal_index(g) = 1;

    x = zeros(n,n);
    x(goal_index == 1) = 1;

    [xg(i), yg(i)] = find(x);
end
l = 3;

ulim = max(J, [], 1);

plt_nr = 3;
plt_nc = 8;
n  = sqrt(size(U, 1));
num_cols = min(48, size(U, 2));
k = 0;
for i=1:3:plt_nr
    for j=1:plt_nc
        k = k+1;
        if k<=num_cols

            subplot(plt_nr,plt_nc, (i-1)*plt_nc + j);
            C = zeros(n,n);
            C(:) = J(:,k);
            imagesc(C', [0, ulim(k)]);
            
            
            set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
            axis image;
            colormap('jet');    
    
            subplot(plt_nr,plt_nc, i*plt_nc + j);
            C = zeros(n,n);
            C(:) = U(:,k);
            imagesc(C', [0, ulim(k)]);
            set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
            axis image;
            colormap('jet');   

            for kg = 1:3
                center = [yg(kg) xg(kg)];
                pos = [center - l/2*[1 1], [l l] ];
                rectangle('Position',pos,'Curvature',[1 1], 'EdgeColor',[1 1 1], 'linewidth', 2);    
                axis equal;
                hold on;
            end


            subplot(plt_nr,plt_nc, (i+1)*plt_nc + j);
            C = zeros(n,n);
            C(:) = corr_map(:,k);
            imagesc(C', [-0 1]);
            set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
            axis image;
            colormap('jet');
            colorbar(gca, 'Location','southoutside');
%             for kg = 1:3
%                 center = [yg(kg) xg(kg)];
%                 pos = [center - l/2*[1 1], [l l] ];
%                 rectangle('Position',pos,'Curvature',[1 1], 'EdgeColor',[1 1 1], 'FaceColor',[1 1 1], 'linewidth', 2);    
%                 axis equal;
%                 hold on;
%             end             
        end
    end
    
end

end

function plot_corr_maps(corr_map, task, plt_nr, plt_nc)
if nargin < 3
    plt_nr = 5;
    plt_nc = 10;
end

goals = task.goals_center;
xy = task.lxy(:, 2:3);
n  = round(sqrt(size(xy, 1)));

for i = 1:length(goals)
    goal = goals{i};
    [~, g] = min(sum((xy - goal).^2, 2));
    goal_index = zeros(n*2, 1);
    goal_index(g) = 1;

    x = zeros(n,n);
    x(goal_index == 1) = 1;

    [xg(i), yg(i)] = find(x);
end
l = 3;

num_cols = min(50, size(corr_map, 2));
k = 0;
for i=1:plt_nr
    for j=1:plt_nc
        k = k+1;
        if k<=num_cols

            subplot(plt_nr,plt_nc, (i-1)*plt_nc + j);
            C = zeros(n,n);
            C(:) = corr_map(:,k);
            imagesc(C', [0 1]);
            set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
            axis image;
            colormap('jet');
            colorbar(gca, 'LOCATION', 'SouthOutside');
     
            for kg = 1:3
                center = [yg(kg) xg(kg)];
                pos = [center - l/2*[1 1], [l l] ];
                rectangle('Position',pos,'Curvature',[1 1], 'EdgeColor',[1 1 1], 'FaceColor',[1 1 1], 'linewidth', 2);    
                axis equal;
                hold on;
            end            
        end
    end
    
end

end

function plot_maps(J, U, task)

goals = task.goals_center;
xy = task.lxy(:, 2:3);
n  = round(sqrt(size(xy, 1)));

for i = 1:length(goals)
    goal = goals{i};
    [~, g] = min(sum((xy - goal).^2, 2));
    goal_index = zeros(n*2, 1);
    goal_index(g) = 1;

    x = zeros(n,n);
    x(goal_index == 1) = 1;

    [xg(i), yg(i)] = find(x);
end

cg = zeros(n, n);
cg(task.goals_index) = 1;

plt_nr = 5;
plt_nc = 2;

umax = 1.05*max(U, [], 1);
umax = [umax umax];

if size(U, 2) == 1
    vectors = [J(:, 1) U(:, 1)];
    umax = [umax(1) umax(1)];

    plt_nr = 3;
else
    vectors = [J(:, 1) U(:, 1) J(:, 2) U(:, 2)];
    umax = [umax(1) umax(1) umax(2) umax(2)];
end

for i=1:2
subplot(plt_nr, plt_nc, i);
activity = ones(n,n);
imagesc(activity, [0 1]);
set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal'); 
axis image;   
hold on;
end

l = 3;
for k = 1:3
    center = [yg(k) xg(k)];
    pos = [center - l/2*[1 1], [l l] ];
    rectangle('Position',pos,'Curvature',[1 1], 'FaceColor',[0 0 0], 'linewidth', 2);    
    axis equal;
    hold on;
end

for j=1:(plt_nr/2)
    for i= 1:plt_nc
        activity = zeros(n,n);
        activity(:) = vectors(:, (j-1)*plt_nc + i);
        clim = [0 umax((j-1)*plt_nc + i)];

        
        n_mat=mulMatCross(zeros(size(activity,1)),zeros (size(activity,1)));
        crosscorr_map = Cross_Correlation(activity, activity, n_mat);
    
    
        subplot(plt_nr, plt_nc, plt_nc + (2*j-2)*plt_nc + i);
        imagesc(activity, clim);        
        set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
%         colorbar;
        colormap('jet')
%         axis equal;
        axis image;        
        hold on;
        
        for k = 1:3
            center = [yg(k) xg(k)];
            pos = [center - l/2*[1 1], [l l] ];
            rectangle('Position',pos,'Curvature',[1 1], 'EdgeColor',[1 1 1], 'linewidth', 2);    
            axis equal;
            hold on;
        end



        subplot(plt_nr,plt_nc, plt_nc + (2*j-1)*plt_nc+i);
        imagesc(crosscorr_map, [-1 1]);
        set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
        colormap('jet')
%         axis equal;
        axis image;
%         colorbar
        hold on;
    end
end

end

function [ma, md, mg, eg] = boccara_fig_data()
% https://apps.automeris.io/wpd/
Dist0= 588.2969891993474;
Dist1= 155.37260318967458;
Dist2= 298.1482557642077;
Dist3= 530.0386841326272;


% .8/Dist3 = x/Dist0;

increase_activity = Dist0/Dist3*.8;

field_pre = Dist1/Dist3*.8;
field_post = Dist2/Dist3*.8;

Dist0= 316.2832413189385;
Dist1= 165.71231663386249;
Dist2= 156.34171767668062;
Dist3= 327.8169236328561;
Dist4= 185.8789625360231;
Dist5= 174.35307356403652;
Dist6= 304.7584496440659;

of_grid = Dist0/Dist6*.6;
post_grid = Dist2/Dist6*.6;

of_grid_error = Dist3/Dist6*.6 - of_grid;
post_grid_error = Dist4/Dist6*.6 - post_grid;

ma = increase_activity;
md = [field_pre field_post];

mg = [of_grid post_grid];
eg = [of_grid_error post_grid_error];

end

