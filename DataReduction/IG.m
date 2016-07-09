% IG.m
% Author: Jinu Jacob
% jj559@drexel.edu
% 7/16
%
% Ranks features by information gain

function IG(fileName)

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

% Compute information gain for each feature

p = size(targets(targets==1),1);       % number of positive events
n = size(targets(targets==-1),1);      % number of negative events
total = p+n;

H = ((-p/total)*log2(p/total))+((-n/total)*log2(n/total));  % overall entropy of dataset; same as 'entropy(data)'
igTable = [];                                               % hold info gain and feature number

for i = 1:size(data,2)                              % iterate through each feature
    uniqueVals = unique(data(:,i));                 % find the unique values
    remainder = 0;
    
    for j = 1:length(uniqueVals)                    % iterate through unique values
        indices = find(data(:,i)==uniqueVals(j));   % find indices in data with value j
        pos = 0; 
        neg = 0;
        
        for k = 1:length(indices)                   % find the target values for each unique value
            if (targets(indices(k)) == 1)
                pos = pos + 1;                      % sum all instances of positive values (+1)
            else
                neg = neg + 1;                      % sum all instances of negative values (-1)
            
            end
        end
        
        k = pos+neg;                                % total instances of unique value
        weight = k/total;                           % weight of value 
        posEnt = 0;                                 % to avoid NaN error when encountering 0*log2(0)
        negEnt = 0;                            
        
        if (pos == 0)                              % zero positive targets
            posEnt = 0;
        else 
            posEnt = (-pos/k)*log2(pos/k);
        end
        
        if (neg == 0)                              % zero negative targets
            negEnt = 0;
        else
            negEnt = (-neg/k)*log2(neg/k);
        end
              
        remainder = remainder + (weight*(posEnt+negEnt));   % sum of unique value entropies * weight
        
    end
    
    ig = H - remainder;         % compute info gain of feature
    igTable(i,:) = [ig, i];     % add to info gain table
    
end

igTable = sortrows(igTable,-1);     % sort table in descending order by info gain
T = table(igTable(:,1),igTable(:,2),'VariableNames',{'InformationGain','FeatureNumber'});
disp(T);
end