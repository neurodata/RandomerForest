clear
close all
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

LineWidth = 0.5;
FontSize = .1;
axWidth = .75;
axHeight = .75;
cbWidth = .1;
cbHeight = axHeight;
axLeft = repmat([FontSize*4.5,FontSize*9+axWidth,FontSize*13.5+2*axWidth],1,5);
axBottom = [(FontSize*19+axHeight*4)*ones(1,3),...
    (FontSize*15+axHeight*3)*ones(1,3),(FontSize*11+axHeight*2)*ones(1,3),...
    (FontSize*7+axHeight)*ones(1,3),(FontSize*3)*ones(1,3)];
figWidth = axLeft(end) + axWidth + FontSize*3;
figHeight = axBottom(1) + axHeight + FontSize*2;

fig = figure;
fig.Units = 'inches';
fig.PaperUnits = 'inches';
fig.Position = [0 0 figWidth figHeight];
fig.PaperPosition = [0 0 figWidth figHeight];
fig.PaperSize = [figWidth figHeight];

ps = [10,100,1000];
d = 1000;
rhos = 1:5;
ntrials = 100;

k = 1;
for i = 1:length(rhos)
    rho = rhos(i);
    for j = 1:length(ps)
        p = ps(j);

        A = zeros(p,d,ntrials);

        for trial = 1:ntrials
            A(:,:,trial) = full(randmat(p,d,'binary-raw',rho/p));
        end

        nnzs = sum(A~=0);

        ax{k} = axes;
        counts{k} = histcounts(nnzs(:),'Normalization','probability');
        bins{k} = 0:length(counts{k})-1;
        hbar = bar(bins{k},counts{k});
        hbar.LineStyle = 'none';
        ax{k}.XLim = [bins{k}(1)-1 bins{k}(end)+1];
        if bins{k}(end) < 10
            ax{k}.XTick = bins{k};
        else
            ax{k}.XTick = 0:3:bins{k}(end);
        end;
%         ax{k}.XTickLabelRotation = 45;
        ax{k}.LineWidth = LineWidth;
        ax{k}.FontUnits = 'inches';
        ax{k}.FontSize = FontSize;
        ax{k}.Units = 'inches';
        ax{k}.Position = [axLeft(k) axBottom(k) axWidth axHeight];
        xlabel('# Variables')
        ylabel('Proportion')
        title(['\rho = ' num2str(rho/p) ', p = ' num2str(p)])
        k = k+1;
    end
end

save_fig(gcf,[rerfPath 'RandomerForest/Figures/nnz_distribution'],{'fig','pdf','png'})