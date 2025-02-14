clear
close all
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

Colors.rf = 'b';
Colors.rerfb = 'g';
Colors.rerfc = 'c';
Colors.frc2 = 'y';
Colors.frc3 = 'k';
Colors.frc4 = 'r';
LineWidth = 2;

load ~/Trunk_vary_n
load('~/RandomerForest/Results/Trunk_bayes_error.mat','bayes_error','dims')
bayes_error(dims~=10&dims~=100) = [];

ntrials = length(TestError{1}.rf);

for j = 1:2
    p = ps(j);
    
    Classifiers = fieldnames(TestError{1,j});

    ErrorMatrix = zeros(ntrials,length(ns),length(Classifiers));
    
    if j==2
        EndIdx = 3;
    else
        EndIdx = length(ns{j});
    end

    for i = 1:EndIdx
        n = ns{j}(i);

        for c = 1:length(Classifiers)
            cl = Classifiers{c};
            ErrorMatrix(:,i,c) = TestError{i,j}.(cl)';
        end
    end

    figure;
    hold on
    for c = 1:length(Classifiers)
        errorbar(ns{j}(1:EndIdx),mean(ErrorMatrix(:,:,c)),std(ErrorMatrix(:,:,c))/sqrt(ntrials),...
            'LineWidth',LineWidth,'Color',Colors.(Classifiers{c}))
    end

    ax = gca;

    ax.XScale = 'log';
%     ax.YScale = 'log';
    ax.FontSize = 16;
    ax.XLim = [10^(log10(min(ns{j}))-0.1) 10^(log10(max(ns{j}))+0.1)];
    ax.YLim = [bayes_error(j)*0.9 max(max(mean(ErrorMatrix)))*1.2];
    ax.XTick = ns{j};
    ax.XTickLabel = cellstr(num2str(ns{j}'))';
    
    hold on
    
    plot([ax.XLim(1),ax.XLim(2)],[bayes_error(j),bayes_error(j)],'LineWidth',LineWidth)

    xlabel('n')
    ylabel('Misclassification Rate')
    title(sprintf('Trunk (p = %d)',p))
    lh = legend('RF','RerF(bin)','RerF(cont)','F-RC(L=2)','F-RC(L=3)','F-RC(L=4)');
    lh.Box = 'off';

    save_fig(gcf,[rerfPath sprintf('RandomerForest/Figures/Trunk_p_%d',p)])
end