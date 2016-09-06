% multiClass.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 8/16
%

function multiClass(fileName)
    
clearvars -except fileName
 
if(nargin < 1)
    fileName = 'CTG.csv';
end

% Read in data
if (exist(fileName, 'file'))
    data = csvread(fileName,2);    % ignore header row and empty row
else
    disp('File not found');
    return;
end

% Randomly swap rows
n = length(data);                           % number of observations
classes = unique(data(:,end));
k = length(classes);
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
training = training(:,1:end-2);      % remove extra training class
training_size = length(training);

r_test = testing(:,end);             % remove testing targets
testing = testing(:,1:end-2);        % remove extra training class
testing_size = length(testing);

m = mean(training);
s = std(training);

training = training - repmat(m,length(training),1);     % subtract mean
training = training./repmat(s,length(training),1);      % element divide by std

testing = testing - repmat(m,length(testing),1);        % standardize testing data
testing = testing./repmat(s,length(testing),1);


% One vs All
predicted = NaN(length(r_test),k);
r_train_i = r_train;

% Train K classifiers
for i=1:k
    % set class instances = 1 and non-class instances = 0
    r_train_i = (r_train == classes(i));
    
    % train classifier 
    svm = fitcsvm(training, r_train_i); 
    predicted(:,i) = svm.predict(testing);
    
end

final_predictions = NaN(size(r_test));
% assign classes based on classifier predictions
for j=1:length(testing)
    ones = find(predicted(j,:));
    if(length(ones)==0)
        final_predictions(j) = randi(k);
    else
        final_predictions(j) = ones(randi(numel(ones)));
    end
end
 
% Calculate accuracy
correct = sum(r_test == final_predictions);
accuracy = correct/length(r_test);

disp(['One vs All Accuracy: ', num2str(accuracy)]);


% One vs One

% Train 1v1 binary classifiers

perms = nchoosek(classes,2);
predicted = NaN(length(r_test),length(perms));
for i=1:length(perms)
    % get training data and targets corresponding to classes
    training_i = training(r_train ==  perms(i,1) | r_train == perms(i,2),:);
    r_train_i = r_train(r_train ==  perms(i,1) | r_train == perms(i,2));
    
    % train classifier
    svm2 = fitcsvm(training_i, r_train_i);
    predicted(:,i) = svm2.predict(testing);
end

% Select classes that appear most
final_predictions = NaN(size(r_test));
for j=1:length(predicted)
    counts = histcounts(predicted(j,:));
    
    % check for ties
    if (range(counts) == 0)
        idx = counts(randi(numel(counts)));
    else
        [dum, idx] = max(counts);
    end
    
    final_predictions(j) = idx;
end

% Calculate accuracy
correct2 = sum(r_test == final_predictions);
accuracy2 = correct2/length(r_test);

disp(['One vs One Accuracy: ', num2str(accuracy2)]);


end
