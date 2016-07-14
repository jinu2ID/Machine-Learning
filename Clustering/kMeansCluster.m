% cluster.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
%
% Performs k-means clustering using just the 6th and 7th features of the data

function kMeansCluster(fileName)

clearvars -except fileName

if(nargin < 1)
    fileName = 'diabetes.csv';
end

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
k = 2;                                      % # of cluster centers
n = size(data,1);                           % # of data instances

% randomly pick k initial reference vectors
rng(0);
idx = randperm(n);
refMat = NaN(k,2);
for i = 1:k
    refMat(i,:) = data(idx(:,i),:);
end

% plot initial setup
figure(1)
plot(data(:,2),data(:,1),'xr');
hold on;
plot(refMat(:,2),refMat(:,1),'ob');
title('Initial Seeds');
legend('instance','seed');

seedsL1Dist = Inf;
iters = 1;
initial = true;
clusterAssg = NaN(n,1);                     % store assigned cluster centers for each data instance
c = {'r','b','y','m','c','r','g','w','k'};  % cluster colors

% repeat until convergence:
while(seedsL1Dist > eps)
    seedsL1Dist = 0;
    
    % assign each observation to nearest cluster (L1 distance)
    for i = 1:n                                       % iterate through each data instance
        min = Inf;
        for j = 1:k                                   % iterate through each cluster center
            dist = norm(data(i,:)-refMat(j,:),1);     % calculate L1 distance between instance and cluster center
            if(dist < min)                              
                min = dist; 
                clusterAssg(i,:) = j;                 % update closest cluster center
            end
        end
    end
    
    % plot initial cluster assignments
    if (initial)
            figure(2);
            hold on;
            for i = 1:k     % reference vectors
                plot(refMat(i,2),refMat(i,1),'marker','o','color',c{i});     
            end
            
            for i = 1:n     % data instances
                plot(data(i,2),data(i,1),'marker','x','color',c{clusterAssg(i,1)});
            end
            title(['Iteration ' num2str(iters)]);
            initial = false;
    end
    
    % compute cluster reference vectors as mean of associated observations
    clusterAvgs = zeros(size(refMat));  
    clusterSize = zeros(k,1);
    
    for i = 1:n             %sum cluster instances
        clusterAvgs(clusterAssg(i,1),:) = clusterAvgs(clusterAssg(i,1),:) + data(i,:);
        clusterSize(clusterAssg(i,1),:) = clusterSize(clusterAssg(i,1),:) + 1;
    end
    
    for i = 1:k
        clusterAvgs(i,:) = clusterAvgs(i,:)./clusterSize(i);    % divide by cluster size
    end
    
    for i = 1:k
        seedsL1Dist = seedsL1Dist + norm(clusterAvgs(i,:)-refMat(i,:),1);   % compute L1 distances between old and new reference vectors
    end
    
    refMat = clusterAvgs;   % update reference vectors
    iters = iters+1;
end

% plot final cluster assignment
figure(3);
hold on;
for i = 1:k
    plot(refMat(i,2),refMat(i,1),'marker','o','color',c{i});
end
for i = 1:n
    plot(data(i,2),data(i,1),'marker','x','color',c{clusterAssg(i,1)});
end
title(['Iteration ' num2str(iters)]);

end
