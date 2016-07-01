% LDA.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
%
% Reduces a dataset to one dimension using linear discriminant analysis

function LDA(fileName)

% Read in data
data = [];
if (exist(fileName, 'file'))
    data = csvread(fileName);
else 
    disp('File not found');
    return;
end

% Standardize data
targets = data(:,1); % save targets
data(:,1) = [];      % remove first column

m = mean(data);
s = std(data);

data = data - repmat(m,size(data,1),1);     % subtract mean
data = data./repmat(s,size(data,1),1);      % element divide by std

% Reduce data to 1D using LDA
class1 = data(targets==-1,:);       % separate data by class
class2 = data(targets==1,:);

m1 = mean(class1);      % compute class means
m2 = mean(class2);

s1 = (size(class1,1) - 1)*cov(class1);   % compute scatter matrices
s2 = (size(class2,1) - 1)*cov(class2);

sW = s1 + s2;       % compute within class scatter matrix
sWInv = inv(sW);    % compute its inverse

mDiff = m1 - m2;    % difference of class means

w = sWInv * transpose(mDiff);   % compute projection matrix

projected = data*w; % project data onto new dimension

% Graph data
figure(1);
plot(projected(targets==-1,1), 0,'ob');
hold on;
plot(projected(targets==1,1), 1,'or');
hold off;
title('Linear Discriminant Analysis');
legend('-1','+1');

end

