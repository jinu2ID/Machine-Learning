% PCA.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 6/16
%
% Performs principal component analysis on a dataset

function PCA(fileName)

clearvars -except fileName

if(nargin < 1)
    fileName = 'diabetes.csv';
end

% Read in data
data = [];
if (exist(fileName, 'file'))
    data = csvread(fileName);
else 
    disp('File not found');
    return;
end

% Standardize data
targets = data(:,1); % save targets for plotting
data(:,1) = [];      % ignore first column

m = mean(data);
s = std(data);

data = data - repmat(m,size(data,1),1);     % subtract mean
data = data./repmat(s,size(data,1),1);      % element divide by std

% Reduce data to 2D using PCA
cv = cov(data);         % compute covariance matrix

[vec,val] = eig(cv);    % compute eigenvector

W = vec(:,end-1:end);                   % 1st and 2nd principle components
projected = data*W;                     % project data onto principle components 

% Graph data
figure(1);
plot(projected(targets==-1,2), projected(targets==-1,1),'xb');
hold on;
plot(projected(targets==1,2), projected(targets==1,1),'or');
hold off;
title('Principal Component Analysis');
xlabel('Principle Component 1');
ylabel('Principle Component 2');
legend('-1','+1');

end