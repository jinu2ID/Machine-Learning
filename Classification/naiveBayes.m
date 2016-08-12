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
n = length(data);                           % number of observations
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

classes = unique(r_train);      % unique class values
numClasses = length(classes);   % number of classes
prior = NaN(numClasses,1);      % prior probability of class

for i=1:numClasses
   prior(i) = sum(double(r_train == classes(i)))/length(r_train);            % calculate the prior for each class 
end


% Create Normal models for each feature for each class
for i=1:numClasses
   class_i = training((r_train==classes(i)),:);               % for each class i
   m_i(i,:) = mean(class_i,1);                                % compute mean for each feature
   std_i(i,:) = std(class_i,1);                               % compute standard deviation for each feature 
end

% Classify testing samples
for i=1:length(testing)                                                      % for each testing sample:
    for j=1:numClasses
        probs(i,j) = prior(j)*prod(normpdf(testing(i,:),m_i(j),std_i(j)));  % calculate probabilities of being in each class
    end
    
    [M,I] = max(probs(i,:));        % classify sample as class with highest probability
    predicted(i,:) = classes(I);    
end

% Compute statistics
TP = 0;
TN = 0;
FP = 0;
FN = 0;

for i=1:length(r_test)
   if (r_test(i)==1)    % actually positive
       if (predicted(i)==1) 
           TP = TP+1;
       else
           FN =FN+1;
       end
   else     % actually negative
       if (predicted(i)==1)
           FP = FP+1;
       else
           TN = TN+1;
       end
   end
end

TP
TN
FP
FN

precision = TP/(TP + FP)
recall = TP/(TP + FN)
fMeasure = (2 * precision * recall)/(precision + recall) 
accuracy = (TP + TN)/(TP + TN + FP + FN)

end

