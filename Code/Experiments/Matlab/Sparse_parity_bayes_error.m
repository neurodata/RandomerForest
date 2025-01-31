%Computes and plots bayes error for Parity as a function of d

clear
close all
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

n = 1000;
dims = [2 5 10 25 50 100];
ntrials = 10;
bayes_err = NaN(ntrials,length(dims));

for trial = 1:ntrials
    fprintf('trial %d\n',trial)

    for i = 1:length(dims)
        d = dims(i);
	fprintf('d = %d\n',d)
        dgood = min(dims,3);
        X = zeros(n,d);
        Sigma = 1/32*eye(d);
        Mu = sparse(n,d);
        for jj = 1:n
            Mu(jj,:) = binornd(1,0.5,1,d);
            X(jj,1:d) = mvnrnd(Mu(jj,:),Sigma);
        end
        nones = sum(Mu(:,1:dgood),2);
        Y = mod(nones,2);
        
        J = size(Mu,1);
        nlogL = zeros(n,J);
        
        for j = 1:J
            for idx = 1:n
                nlogL(idx,j) = ecmnobj(X(idx,:),Mu(j,:),Sigma);
            end
        end

        [Mn,I] = min(nlogL,[],2);
        bayes_err(trial,i) = sum(Y(I)~=Y)/length(Y);
    end
end

bayes_error = mean(bayes_err);
sem_bayes_error = std(bayes_err)/sqrt(ntrials);

save([rerfPath 'RandomerForest/Results/Sparse_parity_bayes_error.mat'],'bayes_error','sem_bayes_error')
