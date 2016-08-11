% KNN.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 8/16
%
% Performs k-Nearest Neighbors classification on data set

function KNN(fileName, k)

clearvars -except fileName k
 
if(nargin < 1)
    fileName = 'spambase.data';
    k = 5;
end

if(nargin < 2)
    k = 5;
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

dist = pdist2(testing, training, 'cityblock');          % matrix of distances betweeen observations

minThreshold = 0;
maxThreshold = 5;

[vals,locs] = sort(dist);

TP = 0; % True Positives
TN = 0; % True Negatives
FP = 0; % False Positives
FN = 0; % False Negatives
    
for i=1:length(r_test)
    classes = r_train(locs(1:k,i));
    if (r_test(i) == 1) % predicted positive
        if (sum(classes==1)>=k) %true positive
            TP = TP+1;
        else
            FN = FN+1;   % false negative
        end
    else % predicted negative
        if(sum(classes==1)>=k)
            FP = FP + 1; % false positive
        else
            TN = TN + 1; % true negative
        end
    end
end
TP
TN
FP
FN
precision = TP/(TP + FP)
recall = TP/(TP + FN)
accuracy = (TP + TN)/(TP + TN + FP + FN)
fMeasure = (2 * precision * recall)/(precision + recall) 

%figure(1)
%plot(Recall, Precision);


end

