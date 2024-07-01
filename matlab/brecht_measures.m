function [C_pearson, C_pearson_plain, C_home_plain, rate1, rate0, U_peak_rate, J_peak_rate] = brecht_measures(task, model)

box_size = 3;

lxy = task.lxy;
U = model.U;
U_plain = model.U_plain;
J = task.J;
home = task.home;
edge = task.idx_edge;

xy = lxy(:, 2:3)';
[C_pearson] = calc_correlation(xy, box_size, edge, U, J);

[C_pearson_plain] = calc_correlation(xy, box_size, edge, U_plain, J);

[C_home_plain] = calc_correlation(xy, box_size, edge, U, U_plain);

index = measure_dist(home, task.xy);

thr = mean(J(:));
normJ = mean(J>thr);

N = size(U, 2);
rate1 = nan(N, length(index));
rate0 = nan(N, length(index));
for i=1:length(index)
    idx = index{i};

    rate1(:, i) = mean((U(idx,:)>thr)./normJ);
    rate0(:, i) = mean((J(idx,:)>thr)./normJ);
end

% 
U_peak_rate = mean(U>thr, 2);
J_peak_rate = mean(J>thr, 2);

U_peak_rate = U_peak_rate/abs(max(U_peak_rate));
J_peak_rate = J_peak_rate/max(J_peak_rate);
end

function [corr_map] = calc_correlation(xy, box_size, edge, U, J)
corr_map = nan(size(size(xy, 2), 1));
for i=1:size(xy, 2)
    dist = xy - xy(:, i);
    sucessors = abs(dist)<=box_size;
    sucessors = sum(sucessors, 1)==2;

    sucessors(edge) = 0;
    
    y = U(sucessors, :); y = y(:);
    z = J(sucessors, :); z = z(:);

    corr_map(i, 1) = corr(y, z);    
    
end
end

function index = measure_dist(home, xy)
ds = 7;
index = cell(1, d);
if home == 2
    for i=1:ds
        idx = 0;

        idx1 = ( (xy(1, :) > 10-i).*(xy(1, :)<=14+i).*xy(2,:)==16+i);
        idx = idx + idx1;

        idx1 = ( (xy(2, :) >(8-i) ).*(xy(2, :)<=(16+i)).*(xy(1,:)==(11-i)));
        idx = idx + idx1;

        idx1 = ( (xy(2, :) > 8-i).*(xy(2, :)<=16+i).*xy(1,:)==14+i);
        idx = idx + idx1;        

        idx1 = ( (xy(1, :) > 10-i).*(xy(1, :)<=14+i).*xy(2,:)==9-i);
        idx = idx + idx1;             
       
        index{i} = find(idx);
    end

elseif home == 1
    for i=1:ds
        idx = 0;

        idx1 = ( (xy(1, :) > 8-i).*(xy(1, :)<=16+i).*xy(2,:)==21-i);
        idx = idx + idx1;    

        idx1 = (xy(1, :) == 9-i).*(xy(2, :)>=21-i);
        idx = idx + idx1;   

        idx1 = (xy(1, :) == 16+i).*(xy(2, :)>=21-i);
        idx = idx + idx1;           

        index{i} = find(idx);
    end
end

end

