% CFLR.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
%
% Performs closed form linear regression on a data set

function CFLR(fileName)

clearvars -except fileName

if(nargin < 1)
    fileName = 'x06Simple.csv';
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

m = mean(training);
s = std(training);

training = training - repmat(m,length(training),1);     % subtract mean
training = training./repmat(s,length(training),1);      % element divide by std

% add additional feature with value 1
extra = ones(length(training),1);
training = [extra, training];

% Compute closed-form solution of linear regression
w = inv(training' * training) * (training' * r_train)

% Apply solution to testing samples
r_test = testing(:,end);                % remove testing targets
testing = testing(:,1:end-1);

testing = testing - repmat(m,length(testing),1);    % standardize testing data
testing = testing./repmat(s,length(testing),1);

extra = ones(length(testing),1);
testing = [extra, testing];                         % add additional feature 

predicted = w' * testing';


% Compute MSE & RMSE
mse = mean((r_test - predicted').^2);
rmse = sqrt(mse)

end





