close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

rng(1);

Color = [0 1 1;0 1 0;1 0 1;1 0 0;0 0 0];

load Sparse_parity_data
ndims = length(dims);
ntrees = 500;
nmixs = 2:6;
NWorkers = 2;

for i = 3:3

    d = dims(i);
    
    if d <= 5
        mtrys = 1:d;
    else
        mtrys = ceil(d.^[0 1 2]);
    end
    
    if d >= 6
        nmixs = 2:6;
    else
        nmixs = 2:d;
    end 
    
    for trial = 1:1

        fprintf('trial %d\n',trial)

        for j = 1:length(mtrys)
            
            mtry = mtrys(j);

            fprintf('mtry = %d\n',mtry)
            
            poolobj = gcp('nocreate');
            if isempty(poolobj)
                parpool('local',NWorkers);
            end
            
            tic;
            rerf = rpclassificationforest(ntrees,X{i}(:,:,trial),...
                Y{i}(:,trial),'sparsemethod','sparse',...
                'nvartosample',mtry,'NWorkers',NWorkers,'Stratified',true);
            
            Lhat(j) = oobpredict(rerf,X{i}(:,:,trial),Y{i}(:,trial),'last');

            [Yhats,err(j,:)] = oobpredict2(rerf,X{i}(:,:,trial),Y{i}(:,trial));
            dv(j) = diversity(Yhats);

        end
    end
end

strength = 1 - mean(err,2);

ax = subplot(3,1,1);
plot(mtrys,strength','LineWidth',2)
xlabel('mtry')
ylabel('strength')
title('Sparse Parity (n = 1000, p = 10)')
ax.XScale = 'log';

ax = subplot(3,1,2);
plot(mtrys,dv,'LineWidth',2)
xlabel('mtry')
ylabel('diversity')
ax.XScale = 'log';

ax = subplot(3,1,3);
plot(mtrys,Lhat,'LineWidth',2)
xlabel('mtry')
ylabel('ensemble error rate')
ax.XScale = 'log';

save_fig(gcf,[rerfPath 'RandomerForest/Figures/Sparse_parity_strength_and_diversity_vs_mtry'])