%% Compute Sparse Parity True Class Posteriors

close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

p = 10;
p_prime = 3;
Sigma = 1/32*ones(1,p);
xmin = -1;
xmax = 1;
ymin = xmin;
ymax = xmax;
npoints = 50;
[x1gv,x2gv] = meshgrid(linspace(xmin,xmax,npoints),linspace(ymin,ymax,npoints));
X1post = x1gv(:);
X2post = x2gv(:);
X3post = -0.5*ones(npoints^2,p-2);
Xpost = [X1post X2post X3post];
n = size(Xpost,1);

allClusters = double(all_binary_sets(p_prime));
allClusters(allClusters==0) = -1;
nClust = size(allClusters,1);
inc = round(nClust*.01);
for i = 1:nClust
    if mod(i,inc)==0
        fprintf('%d %% complete\n',i/inc)
    end
    f_k(:,i) = all(Xpost(:,1:p_prime).*repmat(allClusters(i,:),n,1)>0,2);
end

g = sum(f_k,2);

truth.posteriors(:,2) = sum(f_k(:,logical(mod(sum(allClusters==1,2),2))),2)./g;
truth.posteriors(:,1) = 1 - truth.posteriors(:,2);

save([rerfPath 'RandomerForest/Results/Sparse_parity_true_posteriors.mat'],'p','p_prime','Sigma','Xpost','truth')