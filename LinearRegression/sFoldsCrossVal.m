% SFCV.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
%
% Performs s-folds cross-validation on a data set

function SFCV(fileName, s)

clearvars -except fileName folds

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


% Creates S folds
c = {};             % cell array to hold folds
start = 1;
fold_length = ceil(n/s);
for i=1:s
    tail = start + fold_length - 1;
    if (tail > n)
        tail = start + (mod(n,fold_length)) - 1;
    end
    c{i} = data(start:tail,:);                      % add ith fold to cell array
    start = start + fold_length;
end

c = c';    

idx = 1:s;
sqErrors = [];
% For i=1 to S:
for i=1:s
    % Select fold i as testing data and remaining (S-1) as training data
    testing = c{i};
    training = cell2mat(c(~ismember(idx,i)));
    
    size(testing)
    size(training)
    
    % Standardize the data
    r_train = training(:,end);          % remove training targets
    training = training(:,1:end-1);
    
    r_test = testing(:,end);
    testing = testing(:,1:end-1);       % remove testing targets
    
    m = mean(training);
    s = std(training);
    
    training = training - repmat(m,length(training),1);     % standardize training
    training = training./repmat(s,length(training),1);      % 
    
    testing = testing - repmat(m,length(testing),1);        % standardize testing
    testing = testing./repmat(s,length(testing),1);         % 
    
    % Train close-formed linear regression model
    extra = ones(length(training),1);                       % add additional feature
    training = [extra, training];
    
    w = inv(training' * training) * (training' * r_train)  % compute solution   
    
    extra = ones(length(testing),1);                        % add additoinal feature
    testing = [extra, testing];
    
    predicted = w' * testing';
    
    % Compute squared error
    sqErrors = [sqErrors; ((r_test - predicted').^2)];       % add square errors to vector
    
end

% Compute RMSE of all errors
rmse = sqrt(mean(sqErrors))

end

