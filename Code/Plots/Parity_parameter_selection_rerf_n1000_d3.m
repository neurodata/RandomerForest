close all
clear
clc

n = 1000;
d =3;
ntrees = 1000;
ntrials = 20;
NWorkers = 2;
mtrys = 1:d;
Colors = linspecer(2*length(mtrys),'sequential');
err_rf = zeros(ntrees,length(mtrys),ntrials);
err_rerf = zeros(ntrees,length(mtrys),ntrials);

for trial = 1:ntrials
    
    fprintf('trial %d\n',trial)
    
    X = zeros(n,d);
    %Sigma = 1/8*ones(1,d);
    Sigma = 1/32*ones(1,d);
    %nones = randi(d+1,n,1)-1;
    %Y = mod(nones,2);
    %Ystr = cellstr(num2str(Y));
    Mu = sparse(n,d);
    for j = 1:n
        %onesidx = randsample(1:d,nones(j),false);
        %Mu(j,onesidx) = 1;
        Mu(j,:) = binornd(1,0.5,1,d);
        X(j,:) = mvnrnd(Mu(j,:),Sigma);
    end

    nones = sum(Mu,2);
    Ystr = cellstr(num2str(mod(nones,2)));    
    
    i = 1;
    
    for mtry = mtrys
        
        fprintf('mtry = %d\n',mtry)
        
        if mtry <= 10
            sparsemethod = 'lowk';
        else
            sparsemethod = 'sparse';
        end

        rf = rpclassificationforest(ntrees,X,Ystr,'RandomForest',true,'nvartosample',mtry,'NWorkers',NWorkers,'Stratified',true);
        err_rf(:,i,trial) = oobpredict(rf,X,Ystr,'every');
        
        rerf = rpclassificationforest(ntrees,X,Ystr,'sparsemethod',sparsemethod,'nvartosample',mtry,'NWorkers',NWorkers,'Stratified',true);
        err_rerf(:,i,trial) = oobpredict(rerf,X,Ystr,'every');
        
        i = i + 1;
    end
end

sem_rf = std(err_rf,[],3)/sqrt(ntrials);
sem_rerf = std(err_rerf,[],3)/sqrt(ntrials);

var_rf = var(err_rf,0,3);
var_rerf = var(err_rerf,0,3);

mean_err_rf = mean(err_rf,3);
mean_err_rerf = mean(err_rerf,3);

legend_names = {};

i = 1;

for mtry = mtrys
    h(i) = plot(mean_err_rf(:,i),'Color',Colors(i,:));
    hold on
    legend_names{i} = sprintf('rf (mtry = %d)',mtry);
    i = i + 1;
end

for mtry = mtrys
    h(i) = plot(mean_err_rerf(:,i-length(mtrys)),'Color',Colors(i,:));
    hold on
    legend_names{i} = sprintf('rerf (mtry = %d)',mtry);
    i = i + 1;
end

legend(h,legend_names)
xlabel('# trees')
ylabel('oob error')
title(sprintf('Parity (n=%d, d=%d, ntrials=%d)',n,d,ntrials))

save_fig(gcf,sprintf('Parity_parameter_selection_rerf_n%d_d%d',n,d))

figure(2)

i = 1;

for mtry = mtrys
    h(i) = plot(var_rf(:,i),'Color',Colors(i,:));
    hold on
    i = i + 1;
end

for mtry = mtrys
    h(i) = plot(var_rerf(:,i-length(mtrys)),'Color',Colors(i,:));
    hold on
    i = i + 1;
end

legend(h,legend_names)
xlabel('# trees')
ylabel('var(oob error)')
title(sprintf('Parity (n=%d, d=%d, ntrials=%d)',n,d,ntrials))

save_fig(gcf,sprintf('Parity_parameter_selection_rerf_variance_n%d_d%d',n,d))

figure(3)

i = 1;

for mtry = mtrys
    h(i) = errorbar(1:ntrees,mean_err_rf(:,i),sem_rf(:,i),'Color',Colors(i,:));
    hold on
    i = i + 1;
end

for mtry = mtrys
    h(i) = errorbar(1:ntrees,mean_err_rerf(:,i-length(mtrys)),sem_rerf(:,i-length(mtrys)),'Color',Colors(i,:));
    hold on
    i = i + 1;
end

legend(h,legend_names)
xlabel('# trees')
ylabel('oob error')
title(sprintf('Parity (n=%d, d=%d, ntrials=%d)',n,d,ntrials))
save_fig(gcf,sprintf('Parity_parameter_selection_rerf_errorbar_n%d_d%d',n,d))