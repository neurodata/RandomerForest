close all
clear
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

LineWidth = 2;
FontSize = 14;

load aggregated_results_2017_01_16

ax = axes;
for i = 1:length(Results)
    DatasetName = Results(i).Name(1:regexp(Results(i).Name,'\.mat')-1);
    Classifiers = fieldnames(Results(i).TestError);
    for c = 1:length(Classifiers)
        plot(1:Results(i).Params.(Classifiers{c}).nTrees,...
            Results(i).OOBError.(Classifiers{c})(:,Results(i).BestIdx.(Classifiers{c})),...
            'LineWidth',LineWidth)
        hold on
    end
    xlabel('Number of Trees')
    ylabel('Out-of-Bag Error')
    title(DatasetName)
    ax.LineWidth = LineWidth;
    ax.FontSize = FontSize;
    ax.Box = 'off';
    l = legend('RF','F-RC','RR-RF');
    l.Box = 'off';
    hold off
    save_fig(gcf,[rerfPath 'RandomerForest/Figures/ROFLMAO/Error_vs_nTrees_' DatasetName],'pdf')
end