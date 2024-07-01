function h = plot_bar(nr,nc,sub_plts,mx,ex,labels,mstr,abc,col,yls,bw,alf)

if ~iscell(sub_plts)
    sub_plts = num2cell(sub_plts);
end

if nargin<8, abc = ''; end
if nargin<9, col = []; end
if nargin<10, yls = cell(1,length(mx)); end
if nargin<11, bw = []; end
if nargin<12, alf = def('alf'); end

fs = def('fs');
fn = def('fn');
fsy = def('fsy');
fsA = def('fsA');
xsA = def('xsA');
ysA = def('ysA');

if isempty(abc)
    abc = [];    
end

if isempty(col)
    col = def('col');
end

if isempty(ex)
    ex = cell(1,length(mx));
    for i=1:length(mx)
        ex{i} = zeros(size(mx{i}));
    end    
end

if isempty(bw)
    bw = .03*numel(mx{1});
end

h = nan(1,length(mx));
for i=1:length(mx)
    h(i) = subplot(nr,nc,sub_plts{i});    
    errorbarKxN(mx{i}',ex{i}',labels,col,bw,[],alf);    
    set(gca,'fontsize',fs,'box','off');
%     alpha(alf(i));
    ylabel(mstr{i},'fontsize',fsy);
    ax = ancestor(gca, 'axes');
    xaxes = get(ax,'XAxis');
    set(xaxes,'fontsize',fsy);    
    
    if ~isempty(abc)
        text(xsA,ysA,abc(i),'fontsize',fsA,'Unit','normalized','fontname',fn);
    end
    if ~isempty(yls{i})
        ylim(yls{i});
    end
end

end
