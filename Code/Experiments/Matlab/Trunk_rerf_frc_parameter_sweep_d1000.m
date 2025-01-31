close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

rng(1);

load Trunk_data
ndims = length(dims);
ntrees = 1000;
nmixs = 2:6;
NWorkers = 2;

Lhat.rerf = NaN(ndims,25,ntrials);
Lhat.frc = NaN(ndims,25,ntrials);
trainTime.rerf = NaN(ndims,25,ntrials);
trainTime.frc = NaN(ndims,25,ntrials);

for i = ndims:ndims

    d = dims(i);
    fprintf('d = %d\n',d)
    
    if d <= 5
        mtrys = 1:d;
    else
        mtrys = ceil(d.^[0 1/4 1/2 3/4 1]);
    end
    
    if d >= 6
        nmixs = 2:6;
    else
        nmixs = 2:d;
    end 
    
    for trial = 1:ntrials

        fprintf('trial %d\n',trial)

        for j = 1:length(mtrys)
            
            mtry = mtrys(j);

            fprintf('mtry = %d\n',mtry)
            
            poolobj = gcp('nocreate');
            if isempty(poolobj)
                parpool('local',NWorkers);
            end
    
            for k = 1:length(nmixs)
                nmix = nmixs(k);
                s = nmix/d;
                fprintf('nmix = %d\n',nmix)
                
                tic;
                cl.rerf = rpclassificationforest(ntrees,X{i}(:,:,trial),Y{i}(:,trial),'sparsemethod','sparse','s',s,'nvartosample',mtry,'NWorkers',NWorkers,'Stratified',true);
                trainTime.rerf(i,length(nmixs)*(j-1)+k,trial) = toc;
                Lhat.rerf(i,length(nmixs)*(j-1)+k,trial) = oobpredict(cl.rerf,X{i}(:,:,trial),Y{i}(:,trial),'last');
                
                tic;
                cl.frc = rpclassificationforest(ntrees,X{i}(:,:,trial),Y{i}(:,trial),'sparsemethod','frc','nmix',nmix,'nvartosample',mtry,'NWorkers',NWorkers,'Stratified',true);
                trainTime.frc(i,length(nmixs)*(j-1)+k,trial) = toc;
                Lhat.frc(i,length(nmixs)*(j-1)+k,trial) = oobpredict(cl.frc,X{i}(:,:,trial),Y{i}(:,trial),'last');
            end
        end
    end
end

save([rerfPath 'RandomerForest/Results/Trunk_rerf_frc_parameter_sweep_d1000.mat'],'dims','Lhat','trainTime')