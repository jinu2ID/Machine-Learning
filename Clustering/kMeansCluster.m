% cluster.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
%
% Performs k-means clustering using just the 6th and 7th feature of the data with k=2

function kMeansCluster(fileName)
% Read in data
data = [];
if (exist(fileName, 'file'))
    data = csvread(fileName);
else 
    disp('File not found');
    return;
end

% Standardize data
targets = data(:,1); % save targets
data = data(:,7:8);  % use only features/columns 6 & 7

m = mean(data);
s = std(data);

data = data - repmat(m,size(data,1),1);     % subtract mean
data = data./repmat(s,size(data,1),1);      % element divide by std

% Perform k-means clustering
k = 2

% randomly pick k initial reference vectors
rng(0);
idx = randperm(size(data,1));
refMat = NaN(k,2);
for i =1:k
    refMat(i,:) = data(idx(:,i),:);
end


plot(data(:,2),data(:,1),'xr');
hold on;
plot(refMat(:,2),refMat(:,1),'ob');

% repeat until convergence:

    % assign each observation to nearest cluster (L1 distance)
    
    
    % compute cluster reference vectors mean of associated observations
    
end
