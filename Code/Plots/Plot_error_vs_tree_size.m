clear
close all
clc

fpath = mfilename('fullpath');
frcPath = fpath(1:strfind(fpath,'RandomerForest')-1);

load('purple2green')
Colors.rf = ColorMap(1,:);
Colors.rerf = ColorMap(4,:);
Colors.rr_rf = ColorMap(8,:);
Colors.xgb = ColorMap(10,:);

LineStyles.rf = '-';
LineStyles.rerf = '-';
LineStyles.rr_rf = '-';
LineStyles.xgb = '-';
Markers = {'o','x','s','^'};
MarkerSize = 12;
LineWidth = 2;
FontSize = .2;
axWidth = 2;
axHeight = 2;
axLeft = [FontSize*4,FontSize*8+axWidth];
axBottom = FontSize*2.5*ones(1,2);
legWidth = axWidth;
legHeight = axHeight;
legLeft = axLeft(end) + axWidth*2/3 + FontSize;
legBottom = axBottom(end);
figWidth = legLeft + legWidth + FontSize/2;
figHeight = axBottom(1) + axHeight + FontSize*1.5;
% figWidth = axLeft(end) + axWidth + FontSize;
% figHeight = axBottom(1) + axHeight + FontSize*1.5;

fig = figure;
fig.Units = 'inches';
fig.PaperUnits = 'inches';
fig.Position = [0 0 figWidth figHeight];
fig.PaperPosition = [0 0 figWidth figHeight];
fig.PaperSize = [figWidth figHeight];

%% Plot Sparse parity error vs tree size

load ~/Sparse_parity_vary_n

[nn,np] = size(TestError);

Classifiers = {'rf','rerf','rr_rf','xgb'};

ErrorMatrix = NaN(nn,np,length(Classifiers));
DepthMatrix = NaN(nn,np,length(Classifiers));
ntrials = length(TestError{1}.rf);

for c = 1:length(Classifiers)
    for j = 1:np
        p = ps(j);
        for i = 1:nn
            n = ns{j}(i);
            if ~strcmp(Classifiers{c},'xgb')
                if ~isempty(TestError{i,j}.(Classifiers{c}))
                    ErrorMatrix(i,j,c) = mean(TestError{i,j}.(Classifiers{c}));
                    D = zeros(ntrials,1);
                    for trial = 1:ntrials
                        D(trial) = mean(Depth{i,j}.(Classifiers{c})(trial,:,BestIdx{i,j}.(Classifiers{c})(trial)));
                    end
                    DepthMatrix(i,j,c) = mean(D);
                else
                    ErrorMatrix(i,j,c) = NaN;
                    DepthMatrix(i,j,c) = NaN;
                end
            else
                fh = fopen(sprintf('~/Documents/R/Results/dat/Sparse_parity_vary_n_testError_n%d_p%d.dat',n,p));
                e = textscan(fh,'%f');
                fclose(fh);
                fh = fopen(sprintf('~/Documents/R/Results/dat/Sparse_parity_vary_n_depth_n%d_p%d.dat',n,p));
                D = textscan(fh,'%f');
                fclose(fh);
                if ~isempty(e{1})
                    ErrorMatrix(i,j,c) = mean(e{1});
                    DepthMatrix(i,j,c) = mean(D{1});
                else
                    ErrorMatrix(i,j,c) = NaN;
                    DepthMatrix(i,j,c) = NaN;
                end
            end
        end
    end
end

ax(1) = axes;

for j = 1:np
    for c = 1:length(Classifiers)
        E = ErrorMatrix(:,j,c);
        D = DepthMatrix(:,j,c);
        plot(D,E,Markers{j},'MarkerSize',MarkerSize,'MarkerEdgeColor',...
            Colors.(Classifiers{c}))
        hold on
    end
end

xlabel('Mean Tree Depth')
ylabel('Error Rate')
title('Sparse Parity')
    
ax(1).LineWidth = LineWidth;
ax(1).FontUnits = 'inches';
ax(1).FontSize = FontSize;
ax(1).Units = 'inches';
ax(1).Position = [axLeft(1) axBottom(1) axWidth axHeight];
ax(1).Box = 'off';
% ax(1).XLim = [ns{j}(1) ns{j}(end)];
% ax(1).XScale = 'log';
% ax(1).XTick = ns{j};
% ax(1).YLim = [0.05 0.5];

clear ErrorMatrix DepthMatrix
%% Plot Sparse parity error vs tree size

load ~/Trunk_vary_n

[nn,np] = size(TestError);

Classifiers = {'rf','rerf','rr_rf','xgb'};

ErrorMatrix = NaN(nn,np,length(Classifiers));
DepthMatrix = NaN(nn,np,length(Classifiers));
ntrials = length(TestError{1}.rf);

for c = 1:length(Classifiers)
    for j = 1:np
        p = ps(j);
        for i = 1:nn
            n = ns{j}(i);
            if ~strcmp(Classifiers{c},'xgb')
                if ~isempty(TestError{i,j}.(Classifiers{c}))
                    ErrorMatrix(i,j,c) = mean(TestError{i,j}.(Classifiers{c}));
                    D = zeros(ntrials,1);
                    for trial = 1:ntrials
                        D(trial) = mean(Depth{i,j}.(Classifiers{c})(trial,:,BestIdx{i,j}.(Classifiers{c})(trial)));
                    end
                    DepthMatrix(i,j,c) = mean(D);
                else
                    ErrorMatrix(i,j,c) = NaN;
                    DepthMatrix(i,j,c) = NaN;
                end
            else
                fh = fopen(sprintf('~/Documents/R/Results/dat/Trunk_vary_n_testError_n%d_p%d.dat',n,p));
                e = textscan(fh,'%f');
                fclose(fh);
                fh = fopen(sprintf('~/Documents/R/Results/dat/Trunk_vary_n_depth_n%d_p%d.dat',n,p));
                D = textscan(fh,'%f');
                fclose(fh);
                if ~isempty(e{1})
                    ErrorMatrix(i,j,c) = mean(e{1});
                    DepthMatrix(i,j,c) = mean(D{1});
                else
                    ErrorMatrix(i,j,c) = NaN;
                    DepthMatrix(i,j,c) = NaN;
                end
            end
        end
    end
end

ax(2) = axes;

for j = 1:np
    for c = 1:length(Classifiers)
        E = ErrorMatrix(:,j,c);
        D = DepthMatrix(:,j,c);
        plot(D,E,Markers{j},'MarkerSize',MarkerSize,'MarkerEdgeColor',...
            Colors.(Classifiers{c}))
        hold on
    end
end

xlabel('Mean Tree Depth')
ylabel('Error Rate')
title('Trunk')
    
ax(2).LineWidth = LineWidth;
ax(2).FontUnits = 'inches';
ax(2).FontSize = FontSize;
ax(2).Units = 'inches';
ax(2).Position = [axLeft(2) axBottom(2) axWidth axHeight];
ax(2).Box = 'off';
% ax(1).XLim = [ns{j}(1) ns{j}(end)];
% ax(1).XScale = 'log';
% ax(1).XTick = ns{j};
% ax(1).YLim = [0.05 0.5];

save_fig(gcf,[frcPath 'RandomerForest/Figures/Sparse_parity_error_vs_tree_size'])