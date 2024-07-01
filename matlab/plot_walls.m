function plot_walls(P, lxy, config)
if nargin<3, config = []; end
if isempty(config)
    config.linewidth = 1;
end

p = inputParser;
p.addParameter('linewidth', 1);
p.addParameter('text_states', 0);
p.addParameter('start', []);
p.addParameter('last', []);
p.addParameter('colorful', zeros(size(P, 1), 1));
p.addParameter('linecolor', 'k');
p.addParameter('show_terminals', 0);
p.addParameter('add_grid', 0);

p.parse(config);
config    = p.Results;

% -------------------------------------------------------------------------

W = 1 - (P > 0); 

% Terminals are not walls
terminals = find(diag(P));

N = size(W,1);

ptr_down = zeros(N,1);
ptr_right = zeros(N,1);

for i = 1:N
  xyi = lxy(i,2:3);
  xyir = xyi + [1 0];
  
  j = find(lxy(:,2)==xyir(1) & lxy(:,3)==xyir(2));
  if ~isempty(j)
    j = lxy(j,1);
    if W(i,j) ~= 0
      ptr_down(i) = 1; 
    end
  end

  xyir = xyi + [0 1];
  j = find(lxy(:,2)==xyir(1) & lxy(:,3)==xyir(2));
  if ~isempty(j)
    j = lxy(j,1);
    if W(i,j) ~= 0
      ptr_right(i) = 1;
    end
  end
end

rr = lxy(:,2);
cc = lxy(:,3);

constant = 0.5;
% xlim([-constant + min(cc), constant + max(cc)])
% ylim([-constant + min(rr), constant + max(rr)])

for i = 1:length(ptr_right)
  if ptr_right(i) == 1 
    x = [cc(i)+constant cc(i)+constant];
    y = [rr(i)-constant rr(i)+constant];
    plot(x,y,'LineWidth',config.linewidth, 'color', config.linecolor); hold on;
  end
  
  if ptr_down(i) == 1
    x = [cc(i)-constant cc(i)+constant]; 
    y = [rr(i)+constant rr(i)+constant];
    plot(x,y,'LineWidth',config.linewidth, 'color', config.linecolor); hold on;
  end  

  if config.add_grid
    x = [cc(i)-constant cc(i)+constant]; 
    y = [rr(i)+constant rr(i)+constant];
    plot(x,y,'LineWidth',.5, 'color', .5*[1 1 1]); hold on;

    x = [cc(i)+constant cc(i)+constant];
    y = [rr(i)-constant rr(i)+constant];
    plot(x,y,'LineWidth',.5, 'color', .5*[1 1 1]); hold on;
  end

%   if any(terminals == i)
%     center = [cc(i)-constant rr(i)-constant];
%     width = 1; 
%     height = 1;
%     rectangle('Position',[center width height],'FaceColor','r','EdgeColor','r');  hold on;
%   end

    if config.text_states
        if i<1000
            x = [cc(i)] ;
            y = [rr(i)] ; 
            text(x, y, sprintf('%3d', i), 'HorizontalAlignment', 'center'); hold on;
        end
    end

    if config.colorful(i)
        center = [cc(i)-constant rr(i)-constant];
        width = 1;
        height = 1;  
        rectangle('Position',[center width height],'FaceColor','r','EdgeColor','r'); hold on;     
    end    

    if config.show_terminals
        if any(terminals == i)
            center = [cc(i)-constant rr(i)-constant];
            width = 1;
            height = 1;  
            rectangle('Position',[center width height],'FaceColor','r','EdgeColor','r'); hold on;     
        end
    end
end

if ~isempty(config.start)
  i = config.start;
  center = [cc(i)-constant rr(i)-constant];
  width = 1;
  height = 1;
  rectangle('Position',[center width height],'FaceColor','b','EdgeColor','b') 
end

if ~isempty(config.last)
  i = config.last;
  center = [cc(i)-constant rr(i)-constant];
  width = 1;
  height = 1;  
  rectangle('Position',[center width height],'FaceColor','r','EdgeColor','r')
end  

xlim([-constant + min(cc), constant + max(cc)])
ylim([-constant + min(rr), constant + max(rr)])

set(gca,'XTick',[],'YTick',[], 'box', 'on')
end