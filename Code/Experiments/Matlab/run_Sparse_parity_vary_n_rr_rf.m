close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

rng(1);

ps = [3,10,20];
ns = {[10,100,1000], [100,1000,10000], [1000,5000,10000]};
ntrials = 10;
ntest = 10e3;

Classifiers = {'rr_rf'};

OOBError = cell(length(ns{1}),length(ps));
OOBAUC = cell(length(ns{1}),length(ps));
TrainTime = cell(length(ns{1}),length(ps));
Depth = cell(length(ns{1}),length(ps));
NumNodes = cell(length(ns{1}),length(ps));
NumSplitNodes = cell(length(ns{1}),length(ps));
TestError = cell(length(ns{1}),length(ps));
Bias = cell(length(ns{1}),length(ps));
Variance = cell(length(ns{1}),length(ps));
MR = cell(length(ns{1}),length(ps));
BestIdx = cell(length(ns{1}),length(ps));
Labels = {'0';'1'};

for j = 2:length(ps)
    p = ps(j);
    fprintf('p = %d\n',p)
    
    Xtest = dlmread(sprintf('/scratch/groups/jvogels3/tyler/R/Data/Sparse_parity/dat/Test/Sparse_parity_test_set_p%d.dat',p));
    Ytest = cellstr(num2str(Xtest(:,end)));
    Xtest(:,end) = [];
    ClassPosteriors = dlmread(sprintf('/scratch/groups/jvogels3/tyler/R/Data/Sparse_parity/dat/Test/Sparse_parity_test_set_posteriors_p%d.dat',p));
    
    if p <= 5
        mtrys = [1:p p^2 p^3];
    else
        mtrys = ceil(p.^[1/4 1/2 3/4 1 2 3]); 
    end
    mtrys_rf = mtrys(mtrys<=p);
      
    for i = 1:length(ns{j})
        fprintf('n = %d\n',ns{j}(i))

        for c = 1:length(Classifiers)
            fprintf('%s start\n',Classifiers{c})
            
            if ns{j}(i) <= 1000
                Params{i,j}.(Classifiers{c}).nTrees = 1000;
            else
                Params{i,j}.(Classifiers{c}).nTrees = 500;
            end
            Params{i,j}.(Classifiers{c}).Stratified = true;
            Params{i,j}.(Classifiers{c}).NWorkers = 12;
            Params{i,j}.(Classifiers{c}).Rescale = 'off';
            Params{i,j}.(Classifiers{c}).mdiff = 'off';
            if strcmp(Classifiers{c},'rf')
                Params{i,j}.(Classifiers{c}).ForestMethod = 'rf';
                Params{i,j}.(Classifiers{c}).d = mtrys_rf;
            elseif strcmp(Classifiers{c},'rerf')
                Params{i,j}.(Classifiers{c}).ForestMethod = 'rerf';
                Params{i,j}.(Classifiers{c}).RandomMatrix = 'binary';
                Params{i,j}.(Classifiers{c}).d = mtrys;
            elseif strcmp(Classifiers{c},'rr_rf')
                Params{i,j}.(Classifiers{c}).ForestMethod = 'rf';   
                Params{i,j}.(Classifiers{c}).Rotate = true;
                Params{i,j}.(Classifiers{c}).d = mtrys_rf;
            end

            OOBError{i,j}.(Classifiers{c}) = NaN(ntrials,length(Params{i,j}.(Classifiers{c}).d),Params{i,j}.(Classifiers{c}).nTrees);
            OOBAUC{i,j}.(Classifiers{c}) = NaN(ntrials,length(Params{i,j}.(Classifiers{c}).d),Params{i,j}.(Classifiers{c}).nTrees);
            TrainTime{i,j}.(Classifiers{c}) = NaN(ntrials,length(Params{i,j}.(Classifiers{c}).d));
            Depth{i,j}.(Classifiers{c}) = NaN(ntrials,Params{i,j}.(Classifiers{c}).nTrees,length(Params{i,j}.(Classifiers{c}).d));
            NumNodes{i,j}.(Classifiers{c}) = NaN(ntrials,Params{i,j}.(Classifiers{c}).nTrees,length(Params{i,j}.(Classifiers{c}).d));
            NumSplitNodes{i,j}.(Classifiers{c}) = NaN(ntrials,Params{i,j}.(Classifiers{c}).nTrees,length(Params{i,j}.(Classifiers{c}).d));
            TestError{i,j}.(Classifiers{c}) = NaN(ntrials,1);
            Bias{i,j}.(Classifiers{c}) = NaN(1,length(Params{i,j}.(Classifiers{c}).d));
            Variance{i,j}.(Classifiers{c}) = NaN(1,length(Params{i,j}.(Classifiers{c}).d));
            MR{i,j}.(Classifiers{c}) = NaN(1,length(Params{i,j}.(Classifiers{c}).d));
            TestPredictions = cell(ntest,ntrials,length(Params{i,j}.(Classifiers{c}).d));
            BestIdx{i,j}.(Classifiers{c}) = NaN(ntrials,1);

            for trial = 1:ntrials
                fprintf('Trial %d\n',trial)
                
                Xtrain = dlmread(sprintf('/scratch/groups/jvogels3/tyler/R/Data/Sparse_parity/dat/Train/Sparse_parity_train_set_n%d_p%d_trial%d.dat',ns{j}(i),p,trial));
                Ytrain = cellstr(num2str(Xtrain(:,end)));
                Xtrain(:,end) = [];

                % train classifier
                poolobj = gcp('nocreate');
                if isempty(poolobj)
                    parpool('local',Params{i,j}.(Classifiers{c}).NWorkers,...
                        'IdleTimeout',360);
                end

                [Forest,~,TrainTime{i,j}.(Classifiers{c})(trial,:)] = ...
                    RerF_train(Xtrain,Ytrain,Params{i,j}.(Classifiers{c}));
                
                fprintf('Training complete\n')

                % compute oob auc, oob error, and tree stats

                for k = 1:length(Forest)
                    Scores = rerf_oob_classprob(Forest{k},...
                        Xtrain,'every');
                    for t = 1:Forest{k}.nTrees
                        Predictions = predict_class(Scores(:,:,t),Forest{k}.classname);
                        OOBError{i,j}.(Classifiers{c})(trial,k,t) = ...
                            misclassification_rate(Predictions,Ytrain,...
                        false);
                        if size(Scores,2) > 2
                            Yb = binarize_labels(Ytrain,Forest{k}.classname);
                            [~,~,~,OOBAUC{i,j}.(Classifiers{c})(trial,k,t)] = ... 
                                perfcurve(Yb(:),Scores((t-1)*ns{j}(i)*Params{i,j}.(Classifiers{c}).d+(1:ns{j}(i)*Params{i,j}.(Classifiers{c}).d)),'1');
                        else
                            [~,~,~,OOBAUC{i,j}.(Classifiers{c})(trial,k,t)] = ...
                                perfcurve(Ytrain,Scores(:,2,t),'1');
                        end
                    end
                    Depth{i,j}.(Classifiers{c})(trial,:,k) = forest_depth(Forest{k})';
                    NN = NaN(1,Forest{k}.nTrees);
                    NS = NaN(1,Forest{k}.nTrees);
                    Trees = Forest{k}.Tree;
                    parfor kk = 1:Forest{k}.nTrees
                        NN(kk) = Trees{kk}.numnodes;
                        NS(kk) = sum(Trees{kk}.isbranch);
                    end
                    NumNodes{i,j}.(Classifiers{c})(trial,:,k) = NN;
                    NumSplitNodes{i,j}.(Classifiers{c})(trial,:,k) = NS;
                    Scores = rerf_classprob(Forest{k},Xtest,'last');
                    TestPredictions(:,trial,k) = predict_class(Scores,Forest{k}.classname);
                end
                
                % select best model for test predictions
                BI = hp_optimize(OOBError{i,j}.(Classifiers{c})(trial,:,end),...
                    OOBAUC{i,j}.(Classifiers{c})(trial,:,end));
                BestIdx{i,j}.(Classifiers{c})(trial) = BI(end);
                
                TestError{i,j}.(Classifiers{c})(trial) = ...
                    misclassification_rate(TestPredictions(:,trial,BestIdx{i,j}.(Classifiers{c})(trial)),Ytest,false);

                clear Forest
            end
            for k = 1:length(Params{i,j}.(Classifiers{c}).d)
                Bias{i,j}.(Classifiers{c})(k) = classifier_bias(TestPredictions(:,:,k),ClassPosteriors);
                Variance{i,j}.(Classifiers{c})(k) = classifier_variance(TestPredictions(:,:,k));
                Phats = NaN(ntest,2);
                for l = 1:2
                    isY = NaN(ntest,ntrials);
                    for trial = 1:ntrials
                        isY(:,trial) = strcmp(TestPredictions(:,trial,k),Labels{l});
                    end
                    Phats(:,l) = mean(isY,2);
                end
                MR{i,j}.(Classifiers{c})(k) = mean(1 - sum(Phats.*ClassPosteriors)); % error estimate using the definition from the bias-variance decomposition
            end
            fprintf('%s complete\n',Classifiers{c})
            save([rerfPath 'RandomerForest/Results/Sparse_parity_vary_n_' Classifiers{c} '.mat'],'ps',...
                'ns','Params','OOBError','OOBAUC','TestError',...
                'TrainTime','Depth','NumNodes','NumSplitNodes','Bias',...
                'Variance','MR','BestIdx')
        end
    end   
end