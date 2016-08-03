% gradientDescent.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
% 
% Performs batch gradient descent on a data set

function gradientDescent(fileName)

clearvars -except fileName

if(nargin < 1)
    fileName = 'x06Simple.csv';
    s = 5;
end

% Read in data
if (exist(fileName, 'file'))
    data = csvread(fileName,1,1); % ignore header row and index column
else
    disp('File not found');
    return;
end

% Randomly swap rows
n = length(data);
rng(0);
ordering = randperm(n);
rand_data = data(ordering,:);

% Select 2/3 of data for training
t = ceil(length(rand_data) * (2/3));
training = rand_data(1:t,:);

% Remaining 1/3 for testing
testing = rand_data(t+1:end,:);


% Standardize data
r_train = training(:,end);           % remove training targets
training = training(:,1:end-1);

r_test = testing(:,end);             % remove testing targets
testing = testing(:,1:end-1);

m = mean(training);
s = std(training);

training = training - repmat(m,length(training),1);     % subtract mean
training = training./repmat(s,length(training),1);      % element divide by std

testing = testing - repmat(m,length(testing),1);        % standardize testing data
testing = testing./repmat(s,length(testing),1);

% Add offset feature
extra = ones(length(training),1);
training = [extra, training];
extra = ones(length(testing),1);
testing = [extra, testing];

% Initialize parameters of W using random values
cols = size(training, 2);
w = (2).*rand(cols,1)-1;

oldRMSE = inf;
rmseVec = [];
lrnRate= 0.01;
iter = 0;
% Repeat until convergence
while(1)
    if (iter == 1000000)
        break;
    end
    
    % Calculate training error
    predicted = w' * training';
    rmse = sqrt(mean((r_train - predicted').^2));
    rmseVec = [rmseVec, rmse];
    percentChng = ((oldRMSE - rmse)/oldRMSE) * 100;
    
    if (percentChng < eps)
        break;
    end
    
    % For each parameter
    for j=1:cols
        w(j) = w(j) - ((lrnRate/n)*sum((predicted' - r_train)'*training(:,j))); 
    end
    
    oldRMSE = rmse;
    iter = iter + 1;
end

% Compute the RMSE of the testing data.
predicted = w' * testing';
iters = 0:iter;
plot(iters, rmseVec);
xlabel('Iteration');
ylabel('RMSE of Training Data');
w
rmse = sqrt(mean((r_test - predicted').^2))

end