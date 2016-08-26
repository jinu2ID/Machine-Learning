% builtinClassify.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 8/16
%
% Performs Naives Bayes, Decision Tree, KNN, SVM and ANN classification on
% dataset and prints a table of precision, recall, fmeasure, and accuracy
% for each classifier.

function builtinClassify(fileName)

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

stats_table = NaN(5,4);

% Naive Bayes
nb = NaiveBayes.fit(training, r_train);
[bayes_predicted] = nb.predict(testing);

TP = length(r_test((r_test==1) & (bayes_predicted==1)));
TN = length(r_test((r_test==0) & (bayes_predicted==0)));
FP = length(r_test((r_test==0) & (bayes_predicted==1)));
FN = length(r_test((r_test==1) & (bayes_predicted==0)));
precision = TP/(TP+FP);
recall = TP/(TP+FN);
fmeasure = (2*precision*recall)/(precision+recall);
accuracy = (TP + TN)/(TP+TN+FP+FN);

stats_table(1,:) = [precision, recall, fmeasure, accuracy];


% Decision Tree
dt = fitctree(training, r_train);
[tree_predicted] = dt.predict(testing);

TP = length(r_test((r_test==1) & (tree_predicted==1)));
TN = length(r_test((r_test==0) & (tree_predicted==0)));
FP = length(r_test((r_test==0) & (tree_predicted==1)));
FN = length(r_test((r_test==1) & (tree_predicted==0)));
precision = TP/(TP+FP);
recall = TP/(TP+FN);
fmeasure = (2*precision*recall)/(precision+recall);
accuracy = (TP + TN)/(TP+TN+FP+FN);

stats_table(2,:) = [precision, recall, fmeasure, accuracy];

% K-Nearest Neighbors
kn = ClassificationKNN.fit(training, r_train);
[kn_predicted] = kn.predict(testing);

TP = length(r_test((r_test==1) & (kn_predicted==1)));
TN = length(r_test((r_test==0) & (kn_predicted==0)));
FP = length(r_test((r_test==0) & (kn_predicted==1)));
FN = length(r_test((r_test==1) & (kn_predicted==0)));
precision = TP/(TP+FP);
recall = TP/(TP+FN);
fmeasure = (2*precision*recall)/(precision+recall);
accuracy = (TP + TN)/(TP+TN+FP+FN);

stats_table(3,:) = [precision, recall, fmeasure, accuracy];


% Support Vector Machines
svm = fitcsvm(training, r_train);
[svm_predicted] = svm.predict(testing);

TP = length(r_test((r_test==1) & (svm_predicted==1)));
TN = length(r_test((r_test==0) & (svm_predicted==0)));
FP = length(r_test((r_test==0) & (svm_predicted==1)));
FN = length(r_test((r_test==1) & (svm_predicted==0)));
precision = TP/(TP+FP);
recall = TP/(TP+FN);
fmeasure = (2*precision*recall)/(precision+recall);
accuracy = (TP + TN)/(TP+TN+FP+FN);

stats_table(4,:) = [precision, recall, fmeasure, accuracy];


% Artificial Neural Networks
ann = feedforwardnet(10);
net = train(ann, training', r_train');
[ann_predicted] = net(testing')';
mid = (max(ann_predicted)-min(ann_predicted))/2;
ann_predicted(ann_predicted<0.5)=0;
ann_predicted(ann_predicted>0.5)=1;

TP = length(r_test((r_test==1) & (ann_predicted==1)));
TN = length(r_test((r_test==0) & (ann_predicted==0)));
FP = length(r_test((r_test==0) & (ann_predicted==1)));
FN = length(r_test((r_test==1) & (ann_predicted==0)));
precision = TP/(TP+FP);
recall = TP/(TP+FN);
fmeasure = (2*precision*recall)/(precision+recall);
accuracy = (TP + TN)/(TP+TN+FP+FN);

stats_table(5,:) = [precision, recall, fmeasure, accuracy];

printmat(stats_table, 'Statistics', 'NaiveBayes DecisionTree KNN SVM ANN', 'Precision Recall F-Measure Accuracy');

end