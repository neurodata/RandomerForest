close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

rng(1);

n.train = 100;
n.test = 1000;
dims = [2 10 50 100 500];
ndims = length(dims);
n.trainSets = 20;
Class = [0;1];
X.train = cell(1,ndims);
Y.train = cell(1,ndims);
X.test = cell(1,ndims);
Y.test = cell(1,ndims);
TestPosteriors = cell(1,ndims);

for i = 1:ndims
    d = dims(i);
    d_idx = 1:d;
    mu1 = 1./sqrt(d_idx);
    mu0 = -1*mu1;
    Mu = cat(1,mu0,mu1);
    Sigma = repmat(eye(d),1,1,2);
    obj = gmdistribution(Mu,Sigma);
    x = zeros(n.train,d,n.trainSets);
    y = cell(n.train,n.trainSets);
    for trainSet = 1:n.trainSets
        [x(:,:,trainSet),idx] = random(obj,n.train);
        y(:,trainSet) = cellstr(num2str(Class(idx)));
    end
    X.train{i} = x;
    Y.train{i} = y;
    [X.test{i},idx] = random(obj,n.test);
    Y.test{i} = cellstr(num2str(Class(idx)));
    TestPosteriors{i} = gmm_class_post(X.test{i},Mu,Sigma);
end

save([rerfPath 'RandomerForest/Data/Trunk_partitioned_data.mat'],'X','Y',...
    'TestPosteriors','n','dims')