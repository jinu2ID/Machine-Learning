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
training = training(:,1:end-1);
training_size = length(training);

r_test = testing(:,end);             % remove testing targets
testing = testing(:,1:end-1);
testing_size = length(testing);

m = mean(training);
s = std(training);

training = training - repmat(m,length(training),1);     % subtract mean
training = training./repmat(s,length(training),1);      % element divide by std

testing = testing - repmat(m,length(testing),1);        % standardize testing data
testing = testing./repmat(s,length(testing),1);


% One vs All
r_train_i = NaN(size(r_train));
predicted = NaN(length(r_test),k);

% Train K classifiers
for i=1:k
    r_train_i(r_train ~= classes(i)) = 0;   % change non-class instances to 0   
    r_train_i(r_train == classes(i)) = 1;   % change class instances to 1
    
    % train classifier 
    svm = fitcsvm(training, r_train_i); 
    predicted(:,i) = svm.predict(testing);
    
end

final_predictions = NaN(size(r_test));
% assign classes based on classifier predictions
for j=1:length(testing)
    % predicted multiple classes, select one randomly
    if (sum(predicted(j,:)>1))
        ones = find(predicted(j,:));                        % idx of elements == 1
        final_predictions(j) = ones(randi(numel(ones)));    % random element 
        
    % did not make a prediction, randomly select class
    elseif (sum(predicted(j,:) == 0))
        final_predictions(j) = randi(numel(predicted(j,:)));
        
    % 
    else
        
    end
    
    
end

%final_predictions

end
