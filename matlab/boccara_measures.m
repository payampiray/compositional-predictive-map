function [increase_activity_mean, field_dist_post, field_dist_pre, correlation, attraction, distance, grid_post, grid_pre] = boccara_measures(f)

task = f.task;
model = f.model;
U = model.U;
J = f.J;
gridJ = f.Gridness60';
gridU = model.Gridness60';

goals_index = task.goals_index;

task.thr_field = 0.5;
task.thr_near = 2.5;

goals = task.goals_center;
thr_near = task.thr_near;
xy = task.lxy(:, 2:3);

near_goals_index = 0;
for k=1:3
    goal = goals{k};    
    dist = sqrt(sum((xy - goal).^2, 2));
    
    near_goal = (dist< thr_near);
    near_goals_index = near_goals_index + near_goal;
end
near_goals_index = near_goals_index>0;
near_goals_index = find(near_goals_index);


Ux = U./max(U);
Jx = J./max(J);

[correlation, attraction, distance] = calc_attraction_distance_corr(Ux, Jx, task);

% find the strongest field pre and post
[~, max_post_index] = max(U);
[~, max_pre_index] = max(J);

goals = task.goals_center;
distance_post = nan(length(max_post_index), length(goals));
distance_pre = nan(length(max_post_index), length(goals));
xy = task.lxy(:, 2:3);
for i=1:length(max_post_index)
%     distance_post(i, :) = sqrt(sum((xy(max_post_index(i), :) - goals).^2, 2));
%     distance_pre(i, :) = sqrt(sum((xy(max_pre_index(i), :) - goals).^2, 2));

    distance_post(i, :) = min(sqrt(sum((xy(max_post_index(i), :) - xy(goals_index, :)).^2, 2)));
    distance_pre(i, :) = min(sqrt(sum((xy(max_pre_index(i), :) - xy(goals_index, :)).^2, 2)));    
end

distance_maxfield_post = min(distance_post, [], 2);
distance_maxfield_pre = min(distance_pre, [], 2);

field_dist_post = sum(distance_maxfield_post < task.thr_near, 2);
field_dist_pre = sum(distance_maxfield_pre < task.thr_near, 2);

field_dist_pre = mean(field_dist_pre);
field_dist_post = mean(field_dist_post);

thr = mean(median(Jx));
peak_post = mean(U(near_goals_index, :)>thr, 1)';
peak_pre = mean(J(near_goals_index, :)>thr, 1)';


increase_activity_mean = mean((peak_post - peak_pre) >0);

thr = .5;
idx = gridJ>thr;
grid_pre = mean(gridJ(idx));
grid_post = mean(gridU(idx));
end

function [correlation, attraction, distance] = calc_attraction_distance_corr(U, J, task)

xy = task.lxy(:, 2:3);
goals = task.goals_center;
thr_field = task.thr_field;
thr_near = task.thr_near;

size_base = size(J, 2);

attraction_all = []; 
distance_all = [];
corr_per_goal = nan(length(goals), size_base);
attraction = nan(length(goals), size_base);
distance = nan(length(goals), size_base); 

for g = 1:length(goals)
    goal = goals{g};
    for i=1:size_base    
        fields_index = find(J(:, i)> thr_field);
        distance_per_goal_pre = zeros(1, length(fields_index));
        attraction_per_goal = zeros(1, length(fields_index));
        for j = 1:length(fields_index)
            field_index = fields_index(j);        
            distance_per_goal_pre(j) = sqrt(sum((xy(field_index, :) - goal).^2, 2));
    
            near_j = (sqrt(sum((xy(field_index, :) - xy).^2, 2)) < thr_near);        
            attraction_per_goal(j)  = mean((median(U(near_j, :)) - median(J(near_j, :)))>0);
        end
        corr_per_goal(g, i) = corr( attraction_per_goal', distance_per_goal_pre');

        attraction(g, i) = mean(attraction_per_goal, 2);
        distance(g, i) = mean(distance_per_goal_pre, 2);
    
        attraction_all = cat(2, attraction_all, attraction_per_goal);
        distance_all = cat(2, distance_all, distance_per_goal_pre);  
    end
end

attraction = mean(attraction, 1)';
distance = mean(distance, 1)';

correlation = corr(attraction, distance);

end
