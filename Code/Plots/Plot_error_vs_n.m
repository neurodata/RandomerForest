clear
close all
clc

fpath = mfilename('fullpath');
rerfPath = fpath(1:strfind(fpath,'RandomerForest')-1);

load('purple2green')
Colors.rf = ColorMap(1,:);
Colors.rerf = ColorMap(4,:);
Colors.rr_rf = ColorMap(8,:);
Colors.xgb = ColorMap(10,:);

LineStyles.rf = '-';
LineStyles.rerf = '-';
LineStyles.rr_rf = '-';
LineStyles.xgb = '-';
LineWidth = 2;
FontSize = .2;
axWidth = 2;
axHeight = 2;
axLeft = repmat([FontSize*4,FontSize*8+axWidth],1,3);
axBottom = [(FontSize*7.5+axHeight*2)*ones(1,2),...
    (FontSize*4+axHeight)*ones(1,2),FontSize*2*ones(1,2)];
legWidth = axWidth;
legHeight = axHeight;
legLeft = axLeft(end) + axWidth*2/3 + FontSize;
legBottom = axBottom(end);
figWidth = legLeft + legWidth + FontSize;
figHeight = axBottom(1) + axHeight + FontSize*1.5;
% figWidth = axLeft(end) + axWidth + FontSize;
% figHeight = axBottom(1) + axHeight + FontSize*1.5;

fig = figure;
fig.Units = 'inches';
fig.PaperUnits = 'inches';
fig.Position = [0 0 figWidth figHeight];
fig.PaperPosition = [0 0 figWidth figHeight];
fig.PaperSize = [figWidth figHeight];

%% Plot Sparse Parity

load ~/Sparse_parity_vary_n

[nn,np] = size(TestError);

Classifiers = {'rf','rerf','rr_rf','xgb'};

for j = 1:np
    p = ps(j);
    ax(2*j-1) = axes;
    for c = 1:length(Classifiers)
        for i = 1:nn
            n = ns{j}(i);
            if ~strcmp(Classifiers{c},'xgb')
                if ~isempty(TestError{i,j}.(Classifiers{c}))
                    ErrorMatrix.(Classifiers{c})(i,j) = mean(TestError{i,j}.(Classifiers{c}));
                    SEM.(Classifiers{c})(i,j) = std(TestError{i,j}.(Classifiers{c}))/sqrt(length(TestError{i,j}.(Classifiers{c})));
                else
                    ErrorMatrix.(Classifiers{c})(i,j) = NaN;
                    SEM.(Classifiers{c})(i,j) = NaN;
                end
            else
                fh = fopen(sprintf('~/Documents/R/Results/dat/Sparse_parity_vary_n_testError_n%d_p%d.dat',n,p));
                e = textscan(fh,'%f');
                fclose(fh);
                if ~isempty(e{1})
                    ErrorMatrix.(Classifiers{c})(i,j) = mean(e{1});
                    SEM.(Classifiers{c})(i,j) = std(e{1})/sqrt(length(e{1}));
                else
                    ErrorMatrix.(Classifiers{c})(i,j) = NaN;
                    SEM.(Classifiers{c})(i,j) = NaN;
                end
            end
        end
        errorbar(ns{j},ErrorMatrix.(Classifiers{c})(:,j),SEM.(Classifiers{c})(:,j),...
            'LineWidth',LineWidth,'Color',Colors.(Classifiers{c}));
        hold on
    end
    if j==1
        xlabel('n_{train}')
        ylabel({['\bf{p = ' num2str(p) '}'];'\rm{Error Rate}'})
        text(0.5,1.05,'Sparse Parity','FontSize',16,'FontWeight','bold','Units',...
            'normalized','HorizontalAlignment','center','VerticalAlignment'...
            ,'bottom')
    else
        ylabel(['\bf{p = ' num2str(p) '}'])
    end
    
%     title(['(' char('A'+(2*(j-1))) ')'],'Units','normalized','Position',[0.025 .975],'HorizontalAlignment','left','VerticalAlignment','top')
    ax(2*j-1).LineWidth = LineWidth;
    ax(2*j-1).FontUnits = 'inches';
    ax(2*j-1).FontSize = FontSize;
    ax(2*j-1).Units = 'inches';
    ax(2*j-1).Position = [axLeft(2*j-1) axBottom(2*j-1) axWidth axHeight];
    ax(2*j-1).Box = 'off';
    ax(2*j-1).XLim = [ns{j}(1) ns{j}(end)];
    ax(2*j-1).XScale = 'log';
    ax(2*j-1).XTick = ns{j};
%     ax(2*j-1).XTickLabel = {'2' '5' '10' '20' '40'};
    ax(2*j-1).YLim = [0 .51];
    
end

clear ErrorMatrix SEM
%% Plot Trunk

load ~/Trunk_vary_n

[nn,np] = size(TestError);

Classifiers = {'rf','rerf','rr_rf','xgb'};

for j = 1:np
    p = ps(j);
    ax(2*j) = axes;
    for c = 1:length(Classifiers)
        for i = 1:nn
            n = ns{j}(i);
            if ~strcmp(Classifiers{c},'xgb')
                if ~isempty(TestError{i,j}.(Classifiers{c}))
                    ErrorMatrix.(Classifiers{c})(i,j) = mean(TestError{i,j}.(Classifiers{c}));
                    SEM.(Classifiers{c})(i,j) = std(TestError{i,j}.(Classifiers{c}))/sqrt(length(TestError{i,j}.(Classifiers{c})));
                else
                    ErrorMatrix.(Classifiers{c})(i,j) = NaN;
                    SEM.(Classifiers{c})(i,j) = NaN;
                end
            else
                fh = fopen(sprintf('~/Documents/R/Results/dat/Trunk_vary_n_testError_n%d_p%d.dat',n,p));
                e = textscan(fh,'%f');
                fclose(fh);
                if ~isempty(e{1})
                    ErrorMatrix.(Classifiers{c})(i,j) = mean(e{1});
                    SEM.(Classifiers{c})(i,j) = std(e{1})/sqrt(length(e{1}));
                else
                    ErrorMatrix.(Classifiers{c})(i,j) = NaN;
                    SEM.(Classifiers{c})(i,j) = NaN;
                end
            end
        end
        errorbar(ns{j},ErrorMatrix.(Classifiers{c})(:,j),SEM.(Classifiers{c})(:,j),...
            'LineWidth',LineWidth,'Color',Colors.(Classifiers{c}));
        hold on
    end
    
    if j==1
        text(0.5,1.05,'Trunk','FontSize',16,'FontWeight','bold','Units',...
            'normalized','HorizontalAlignment','center','VerticalAlignment'...
            ,'bottom')
    end
    
    ylabel(['\bf{p = ' num2str(p) '}'])
    
%     title(['(' char('A'+2*j-1) ')'],'Units','normalized','Position',[0.025 .975],'HorizontalAlignment','left','VerticalAlignment','top')
    ax(2*j).LineWidth = LineWidth;
    ax(2*j).FontUnits = 'inches';
    ax(2*j).FontSize = FontSize;
    ax(2*j).Units = 'inches';
    ax(2*j).Position = [axLeft(2*j) axBottom(2*j) axWidth axHeight];
    ax(2*j).Box = 'off';
    ax(2*j).XLim = [ns{j}(1) ns{j}(end)];
    ax(2*j).XScale = 'log';
    ax(2*j).XTick = ns{j};
%     ax(2*j).XTickLabel = {'2' '5' '10' '20' '40'};
    if j<=2
        ymax = 0.2;
    else
        ymax = 0.4;
    end
    ax(2*j).YLim = [0 ymax];
    
    if j==np
        [lh,objh] = legend('RF','RerF','RR-RF','XGBoost');
        lh.Box = 'off';
        lh.FontSize = 14;
        lh.Units = 'inches';
        lh.Position = [legLeft legBottom legWidth legHeight];
        for k = 5:length(objh)
            objh(k).Children.Children(2).XData = [(objh(k).Children.Children(2).XData(2)-objh(k).Children.Children(2).XData(1))*.75+objh(k).Children.Children(2).XData(1),objh(k).Children.Children(2).XData(2)];
        end
    end
end

save_fig(gcf,[rerfPath 'RandomerForest/Figures/Simulations_error_vs_n'])