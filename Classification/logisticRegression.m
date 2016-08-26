% logisticRegression.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 8/16
%
% Trains a logistic regression classifier using gradient ascent and plots
% the mean log likelihood 

function logisticRegression(fileName)

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

% Add bias feature
extra = ones(training_size,1);
training = [extra, training];
extra = ones(testing_size,1);
testing = [extra, testing];

% Initialize parameters of W using random values
numFeatures = size(training, 2);
w = (2).*rand(numFeatures,1)-1;

% Train a logistic regression classifier using gradient ascent
learn_rate = 0.5;
iter = 1;
mll = NaN(10000,1);

while(iter < 10001)
    
    % update weights
    for j=1:numFeatures
        % calculate Gw(x)
        GwX = 1.0 ./(1.0 + exp(-training*w));
        w(j) = w(j) + ((learn_rate/n) * sum((r_train - GwX)' * training(:,j)));
    end
    
    % calculate mean log likelihood
%     p1 = NaN(length(training),1);
%     p2 = NaN(length(training),1);
%     for t=1:training_size
%         
%         % avoiding ln(0)
%         p1(t) = log(GwX(t));
%         if (p1(t) == -inf)
%             p1(t) = 0;
%         end
%         p2(t) = log(1-GwX(t));
%         if (p2(t) == -inf)
%             p2(t) = 0;
%         end
%     end

    p1 = log(GwX);
    p1(p1 == -inf) = 0;
    p2 = log(1-GwX);
    p2(p2 == -inf) = 0;
    
    mll(iter) = sum((r_train'*p1) + (1 - r_train)'*p2)/training_size;
    
     
    
    % Terminate when percent change in log likelihood is less than 0.0001
    if (iter ~= 1)
       %iter
       percent_change = abs(mll(iter) - mll(iter-1));
       if (percent_change < eps)
           break
       end
       
       % Decrease learning rate by half if log likelihood decreases
       if (mll(iter) > mll(iter-1))
            %learn_rate = learn_rate/2;
       end
    
    end
    
    iter = iter + 1;
    
end

plot(mll);
mll

end

