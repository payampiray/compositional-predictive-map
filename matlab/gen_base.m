function config = gen_base(config)
% fname = fullfile(sprintf('nnpcas_NmEC%d_gaussian_Size%0.2f.mat', Struct.NmEC, Struct.PcSize ));
if nargin<1, config = []; end
if isempty(config)
    config.fname = 'temp.mat';
end

p = inputParser;
p.addParameter('fname', 'temp.mat');
p.addParameter('NmEC', 50);
p.addParameter('maxx', 10);
p.addParameter('maxy', 10);
p.addParameter('Nx', 25);
p.addParameter('Ny', 25);
p.addParameter('placeCellType', 'Gaussian');
p.addParameter('PcSize', .5);
p.addParameter('rng_seed', 0);
p.addParameter('simdur', 5e5);
p.addParameter('paint', 1);
p.addParameter('arragmentPC', 'array');
p.addParameter('centers', []);
p.addParameter('initial_J', []);
p.addParameter('step_cost', []);

p.parse(config);
config    = p.Results;

if exist(config.fname, 'file')
    config = load(config.fname);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Single output neural network simulaion
%This is the MAIN function receiving place-cell like input and based on PCA
%like architeture caclulates the output.
%Every network has N1 inputs and NmEC outputs which are independent of
%each-other.
%The main reason for differences between outputs are due to the random
%weights assigned to them initially.
%Yedidyah Dordek, Technion 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('functions');
%%  Parameters of simulation
% Struct.simdur =5e5; % total simulation time
config.Resolution = 75;       %resolution of space, used for determing the velocity
%% Model parameters
% number of grid cells
% Struct.NmEC = 1; %for Modules must be more than 1!!!
% Struct.NmEC = 50; %I changed this!

% % number of place cells 
% Struct.Nx = 25; %number of place cells in one dimension (if not in array this is an avarge)
% Struct.Ny = 25; %number of place cells in one dimension (if not in array this is an avarge)
config.N1 = config.Nx*config.Ny; % If a non-arrayed input place field is used, its necessary to have suffeicant amount of Place-cells

% learning rate for plasticity from place to grid cells - inital rate. it
% should cool off and lower in time
config.epsilon = 1e8;   %the actual learning rate is 1/(t*delta+epsilon)
%cooling parmeter see above.
config.delta = 1;
% max weight to a grid cell
config.maxWeight = 0.04; %you don't want to give an initial wight with a too high value

%% Sizes of environment
config.minx = 0; %config.maxx =10; %distance in arbitrary units
config.miny = 0; %config.maxy =10;%distance in arbitrary units
%%  Veclocity
config.vel  =2.15*config.maxx/ config.Resolution;
%should reflect the #Place-cells per distance unit. Needs to ~ maxX/(2*N)

% angular velocity. 
% Note: when using inputs of zero mean (like disks or DOGs) use a small angular velocity in order to get a smoother trajectory. 
% When using Gaussian input +direvation, use a large (2*pi) angular velocity for the sake of smoothness in temporal diffrentiation.
config.angular = .2;  
%% Type of input
%1. Diff of Gaussians
% Struct.placeCellType = 'DOG'; % original
% Struct.placeCellType = 'Gaussian';

%2. Disks
%Struct.placeCellType = 'Disk';

%3. Gaussians
%Struct.placeCellType = 'Gaussian'; %in this case, need to determine the method of mean zero... (adaptation/diffrentioation)
%% Place-cells properties
% choose the arragnment of place cells. in an array or scattered. NOTE: if
% Struct.PcSize = .75; %NOTE!!:when using disks, you need larger sizes. at least 1.5 
% Struct.PcSize = 1.5; %NOTE!!:when using disks, you need larger sizes. at least 1.5 
% Struct.PcSize = .5;

% scattered, you'll need more place-cells in a given environment.
% config.arragmentPC = 'array';
%NOTE: in order for the scattered option to work well (specifically in 
%the case of non restricted weights you need to increase place cell's
%density (and adjust velocity as well...)

% Struct.arragmentPC = 'scattered';

% peak (saturating) value of grid cell activation
config.psiSat = 30;
%% MeanZero method (only for option #3 - Gaussians)
config.meanZaro = 'adpat'; %adaptation
    % growth rate for onset of adaptation, if used.
    config.b1 = 0.5;
    % growth rate of inactivation of adaptation, if used.
    config.b2 =config.b1/3;

%derivatives
%Struct.meanZaro = 'diff'; %diffrentioations

%% Architecture
%single output
config.Arc = 'single';

%multiple output
%Struct.Arc = 'multiple'; %using the symmetric Foldiak's algorithm

%Assymetric architecure
%NOTE: in order for ALL outputs to converge, simulation time needs to be
%long. at least 5e6...
%Struct.Arc = 'sanger'; % using Sanger's algorithm. capable of calc all PCa

%% Output type
%define output fucntion to be sigmoid or linear, NOTE: a large slope (see
%in function) in the sigmoid can cause a quick freeze.

% signoid output
%Struct.output = 'sigmoid';
% Linear output. NOTE: use a rather large slope - help it converge. 
config.output = 'linear';
%% Sending all data to function
%activate an interactive plot of the online activity (1 set of temproal weights, avg activity and weights spatial activity)
paint = config.paint;
%Constrain weights to be nonNegative?
config.NonNegativity =1;
% Save input for PCA? 
%NOTE: the longer the simulation -> more data needed.
% with 16GB  of memory duration of simulation should not exceed 4e5, with 100 outputs, 900 inputs.
config.saveInput = 1;
if config.saveInput
%Allocate memory to speed up process
  config.TemporalInput = zeros(config.N1,config.simdur);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%send the data for simulation !!!!
% fname = fullfile('struct_temporal_temp.mat');
fname = config.fname;
if ~exist(fname, 'file')
rng(config.rng_seed);
[config] = PCANetworkFunc(config, paint);    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PCA and Spatial Maps
[config] = PCAEvecCalc(  config,config.TemporalInput,config.NmEC);
%clear  Struct.TemporalInput  %clear some memory if needed...

config = rmfield(config, 'TemporalInput');
save(fname, '-struct', 'config');

else
config = load(fname);

end

%calculate the Maps, Gridness, angles, etc.
%weightsMap holds the var requested to calc.

weightsMap=config.J; %J are weights of the network
%weightsMap =Struct.PCANNEvec; %calc for NonNegative PCA
%weightsMap =Struct.PCAEvec; %calc for "regular" PCA


%in order to plot a map you need to send in 
 [config.outputMap, config.autocorrMap, config.Gridness60,...
     config.Gridness90, config.minAngles, config.GridSpacing] = Maps(config,weightsMap);
save(fname, '-struct', 'config');

 %% Plot results & Autocorrelations
if paint 
numberOfPlots=2;
n_mat=mulMatCross(zeros(size(config.outputMap,1)),zeros (size(config.outputMap,1)));
plotResults( config.outputMap,n_mat,numberOfPlots)
end
 

end