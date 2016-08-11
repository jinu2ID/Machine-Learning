% naiveBayes.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 8/16
%
% Builds a Naive Bayes Classifier

function naiveBayes(fileName)

clearvars -except fileName
 
if(nargin < 1)
    fileName = 'spambase.data';
end

% Read in data
if (exist(fileName, 'file'))
    data = csvread(fileName);
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

% Divide the training data into two groups: Spam samples, Non-Spam samples.
spam_training = (training(r_train==1,:));
nonspam_training = (training(r_train==0,:));

% Create Normal models for each feature for each class.z
numFeatures = size(spam_training,2);        % the number of columns/features

spam_means = mean(spam_training, 1);        % get means of each feature in spam
spam_std = std(spam_training,1);            % get standard deviation of each feature in spam

nonspam_means = mean(nonspam_training, 1);        % get means of each feature in nonspam
nonspam_std = std(nonspam_training,1);            % get standard deviation of each feature in nonspam




% Classify testing samples using models
end

