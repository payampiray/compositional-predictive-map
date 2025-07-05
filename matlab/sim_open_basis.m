function sim_open_basis()

f = get_base();
lxy = f.lxy;
J = f.J;

idx_main = [19 31:39];

close all;
fsiz = [0 0 .3 .45];
figure; set(gcf,'units','normalized'); set(gcf,'position',fsiz);
plot_maps(J(:, idx_main), lxy);

end

function f = get_base(seed)
if nargin<1, seed = 0; end

fmain = fullfile('sum', 'open_basis.mat');

if ~exist(fmain, 'file')
    PcSize = 0.35;
    addpath('functions');
    
    fdir = fullfile('..','analysis', 'base', sprintf('Size%0.2f_Nx25_Ny25', PcSize));
    fname = fullfile(fdir, sprintf('seed%03d.mat', seed));
    
    f = load(fname);
    xy = f.placeCenters;       
    [~, lxy] = get_xy2P(xy);    
    f.lxy = lxy;
    
    save(fmain, '-struct', 'f');
end
f = load(fmain);
end

% -------------------------------------------------------------------------
function plot_maps(J, lxy)

xy = lxy(:, 2:3);
n  = round(sqrt(size(xy, 1)));

plt_nc = 5;
K = (ceil(size(J, 2)/plt_nc));
plt_nr = 2*K;


umax = 1.05*max(J, [], 1);

for i= 1:size(J,2)
    activity = zeros(n,n);
    activity(:) = J(:, i);
    clim = [0 umax(i)];

    
    n_mat=mulMatCross(zeros(size(activity,1)),zeros (size(activity,1)));
    crosscorr_map = Cross_Correlation(activity, activity, n_mat);

%     k = floor((i-1)/plt_nc)*plt_nc+i;
    k = i;

    subplot(plt_nr, plt_nc, k);
    imagesc(activity, clim);        
    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
%         colorbar;
    colormap('jet')
%         axis equal;
    axis image;        
    hold on;

%     k = ceil(i/plt_nc)*plt_nc+i;
    k = ceil(size(J,2)/plt_nc)*plt_nc + i;
    subplot(plt_nr,plt_nc, k);
    imagesc(crosscorr_map, [-1 1]);
    set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
    colormap('jet')
%         axis equal;
    axis image;
%         colorbar
    hold on;
end

end

function plot_all(U)

plt_nr = 8;
plt_nc = 10;
n  = sqrt(size(U, 1));
num_cols = min(52, size(U, 2));

umax = 1.05*max(U, [], 1);

k = 0;
for i=1:2:plt_nr
    for j=1:plt_nc
        k = k+1;
        if k<=num_cols

            subplot(plt_nr,plt_nc, (i-1)*plt_nc + j);            
            activity = zeros(n,n);
            activity(:) = U(:, k);
            n_mat=mulMatCross(zeros(size(activity,1)),zeros (size(activity,1)));
            crosscorr_map = Cross_Correlation(activity, activity, n_mat); 

            imagesc(activity, [0, umax(k)]);
            set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
            colormap('jet')
            axis image;        
            hold on;
                      
    
            subplot(plt_nr,plt_nc, i*plt_nc + j);
            imagesc(crosscorr_map, [-1 1]);
            set(gca,'xtick',[],'ytick',[],'box','on', 'YDir', 'Normal');    
            colormap('jet')
            axis image;
            hold on;   
        end
    end
    
end

end
