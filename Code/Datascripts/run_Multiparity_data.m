close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

rng(1);

ns = [50,100,500,1000];
ntest = 10000;
p = 15;
ntrials = 10;
p_prime1 = 1:2;
p_prime2 = 3:p;
Sigma1 = 1/64*ones(1,length(p_prime1));
Sigma2 = ones(1,length(p_prime2));
Lambda = [0.67 0.33];
Mu2 = [-0.5*ones(1,length(p_prime2));0.5*ones(1,length(p_prime2))];
obj = gmdistribution(Mu2,Sigma2,Lambda);

for i = 1:length(ns)
    ntrain = ns(i);
    fprintf('ntrain = %d\n',ntrain)
    Xtrain{i} = zeros(ntrain,p,ntrials);
    Ytrain{i} = cell(ntrain,ntrials);
    for trial = 1:ntrials
        Mu1 = sparse(ntrain,length(p_prime1));
        xparity = zeros(ntrain,length(p_prime1));
        for j = 1:ntrain
            Mu1(j,:) = binornd(1,0.5,1,length(p_prime1));
            xparity(j,:) = mvnrnd(Mu1(j,:),Sigma1);
        end
        nOnes = sum(Mu1,2);
        [xgaussian,idx] = random(obj,ntrain);
        Xtrain{i}(:,:,trial) = [xparity,xgaussian];
        y = zeros(ntrain,1);
        y(mod(nOnes,2)==0 & idx==1) = 1;
        y(mod(nOnes,2)==1 & idx==1) = 2;
        y(mod(nOnes,2)==0 & idx==2) = 3;
        y(mod(nOnes,2)==1 & idx==2) = 3;
        Ytrain{i}(:,trial) = cellstr(num2str(y));
    end

    Mu1 = sparse(ntest,length(p_prime1));
    xparity = zeros(ntest,length(p_prime1));
    for j = 1:ntest
        Mu1(j,:) = binornd(1,0.5,1,length(p_prime1));
        xparity(j,:) = mvnrnd(Mu1(j,:),Sigma1);
    end
    nOnes = sum(Mu1,2);
    [xgaussian,idx] = random(obj,ntest);
    Xtest{i} = [xparity,xgaussian];
    y = zeros(ntest,1);
    y(mod(nOnes,2)==0 & idx==1) = 1;
    y(mod(nOnes,2)==1 & idx==1) = 2;
    y(mod(nOnes,2)==0 & idx==2) = 3;
    y(mod(nOnes,2)==1 & idx==2) = 3;
    Ytest{i} = cellstr(num2str(y));
end

save('~/Documents/MATLAB/Data/Multiparity_data.mat','Xtrain','Ytrain',...
    'Xtest','Ytest','ns','ntest','p','p_prime1','p_prime2','ntrials')