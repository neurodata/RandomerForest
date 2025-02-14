%% Plot Performance Profiles for Benchmark Transformations
close all
clear
clc

C = [0 1 1;0 1 0;1 0 1;1 0 0;0 0 0;1 .5 0];
Colors.rf = C(1,:);
Colors.rerf = C(2,:);
Colors.rf_rot = C(3,:);
Colors.rerfr = C(4,:);
Colors.frc = C(5,:);
Colors.rerfdn = C(6,:);

Transformations = {'Untransformed' 'Rotated' 'Scaled' 'Affine'};

runSims = false;

load('~/Benchmarks/Benchmark_untransformed.mat')

idx = setdiff(1:length(Lhat.rf),find(all(isnan(Lhat.rf),2)));

% Only keep frc results for nmix = 2
for i = idx
    nn = sum(~isnan(Lhat.frc(i,:)));
    if nn == 2
        mtrys = 1:2;
        nmixs = 2;
    elseif nn == 6
        mtrys = 1:3;
        nmixs = 2:3;
    elseif nn == 12
        mtrys = 1:4;
        nmixs = 2:4;
    elseif nn == 20
        mtrys = 1:5;
        nmixs = 2:5;
    else
        mtrys = 1:5;
        nmixs = 2:6;
    end
    for j = 1:length(mtrys)
        mtry = mtrys(j);
        for k = 1:length(nmixs)
            nmix = nmixs(k);
            if nmix ~= 2;
                Lhat.frc(i,length(nmixs)*(j-1)+k,:) = NaN;
            end
        end
    end
end

ClRanks = floor(tiedrank([min(Lhat.rf,[],2),min(Lhat.rerf,[],2),...
    min(Lhat.rf_rot,[],2),min(Lhat.rerfr,[],2),min(Lhat.frc,[],2)]')');

Classifiers = {'rf' 'rerf' 'rf_rot' 'rerfr' 'frc'};

RankCounts = NaN(5,length(Classifiers));
for i = 1:5
    RankCounts(i,:) = sum(ClRanks==i);
end

barh(flipud(RankCounts))
ax = gca;
Bars = allchild(ax);
for i = 1:length(Bars)
    Bars(i).EdgeColor = 'w';
    Bars(i).BarWidth = 1;
end

xlabel('Frequency')

title('Untransformed')

ax.FontSize = 14;
ax.YTickLabel = {'5' '4' '3' '2' '1'};

l = legend('RF','RerF','RR-RF','RerF(r)','F-RC');
l.Location = 'northeast';
l.Box = 'off';

save_fig(gcf,'~/Benchmarks/Classifier_ranks_untransformed_horizontal')

load('~/Benchmarks/Benchmark_rotate.mat')

idx = setdiff(1:length(Lhat.rf),find(all(isnan(Lhat.rf),2)));

% Only keep frc results for nmix = 2
for i = idx
    nn = sum(~isnan(Lhat.frc(i,:)));
    if nn == 2
        mtrys = 1:2;
        nmixs = 2;
    elseif nn == 6
        mtrys = 1:3;
        nmixs = 2:3;
    elseif nn == 12
        mtrys = 1:4;
        nmixs = 2:4;
    elseif nn == 20
        mtrys = 1:5;
        nmixs = 2:5;
    else
        mtrys = 1:5;
        nmixs = 2:6;
    end
    for j = 1:length(mtrys)
        mtry = mtrys(j);
        for k = 1:length(nmixs)
            nmix = nmixs(k);
            if nmix ~= 2;
                Lhat.frc(i,length(nmixs)*(j-1)+k,:) = NaN;
            end
        end
    end
end

ClRanks = floor(tiedrank([min(Lhat.rf,[],2),min(Lhat.rerf,[],2),...
    min(Lhat.rf_rot,[],2),min(Lhat.rerfr,[],2),min(Lhat.frc,[],2)]')');

Classifiers = {'rf' 'rerf' 'rf_rot' 'rerfr' 'frc'};

RankCounts = NaN(5,length(Classifiers));
for i = 1:5
    RankCounts(i,:) = sum(ClRanks==i);
end

figure
barh(flipud(RankCounts))
ax = gca;
Bars = allchild(ax);
for i = 1:length(Bars)
    Bars(i).EdgeColor = 'w';
    Bars(i).BarWidth = 1;
end

xlabel('Frequency')

title('Rotated')

ax.FontSize = 14;
ax.YTickLabel = {'5' '4' '3' '2' '1'};

l = legend('RF','RerF','RR-RF','RerF(r)','F-RC');
l.Location = 'northeast';
l.Box = 'off';

save_fig(gcf,'~/Benchmarks/Classifier_ranks_rotated_horizontal')

load('~/Benchmarks/Benchmark_scale2.mat')

idx = setdiff(1:length(Lhat.rf),find(all(isnan(Lhat.rf),2)));

% Only keep frc results for nmix = 2
for i = idx
    nn = sum(~isnan(Lhat.frc(i,:)));
    if nn == 2
        mtrys = 1:2;
        nmixs = 2;
    elseif nn == 6
        mtrys = 1:3;
        nmixs = 2:3;
    elseif nn == 12
        mtrys = 1:4;
        nmixs = 2:4;
    elseif nn == 20
        mtrys = 1:5;
        nmixs = 2:5;
    else
        mtrys = 1:5;
        nmixs = 2:6;
    end
    for j = 1:length(mtrys)
        mtry = mtrys(j);
        for k = 1:length(nmixs)
            nmix = nmixs(k);
            if nmix ~= 2;
                Lhat.frc(i,length(nmixs)*(j-1)+k,:) = NaN;
            end
        end
    end
end

ClRanks = floor(tiedrank([min(Lhat.rf,[],2),min(Lhat.rerf,[],2),...
    min(Lhat.rf_rot,[],2),min(Lhat.rerfr,[],2),min(Lhat.frc,[],2)]')');

Classifiers = {'rf' 'rerf' 'rf_rot' 'rerfr' 'frc'};

RankCounts = NaN(5,length(Classifiers));
for i = 1:5
    RankCounts(i,:) = sum(ClRanks==i);
end

figure
barh(flipud(RankCounts))
ax = gca;
Bars = allchild(ax);
for i = 1:length(Bars)
    Bars(i).EdgeColor = 'w';
    Bars(i).BarWidth = 1;
end

xlabel('Frequency')

title('Scaled')

ax.FontSize = 14;
ax.YTickLabel = {'5' '4' '3' '2' '1'};

l = legend('RF','RerF','RR-RF','RerF(r)','F-RC');
l.Location = 'northeast';
l.Box = 'off';

save_fig(gcf,'~/Benchmarks/Classifier_ranks_scaled_horizontal')

load('~/Benchmarks/Benchmark_affine2.mat')

idx = setdiff(1:length(Lhat.rf),find(all(isnan(Lhat.rf),2)));

% Only keep frc results for nmix = 2
for i = idx
    nn = sum(~isnan(Lhat.frc(i,:)));
    if nn == 2
        mtrys = 1:2;
        nmixs = 2;
    elseif nn == 6
        mtrys = 1:3;
        nmixs = 2:3;
    elseif nn == 12
        mtrys = 1:4;
        nmixs = 2:4;
    elseif nn == 20
        mtrys = 1:5;
        nmixs = 2:5;
    else
        mtrys = 1:5;
        nmixs = 2:6;
    end
    for j = 1:length(mtrys)
        mtry = mtrys(j);
        for k = 1:length(nmixs)
            nmix = nmixs(k);
            if nmix ~= 2;
                Lhat.frc(i,length(nmixs)*(j-1)+k,:) = NaN;
            end
        end
    end
end

ClRanks = floor(tiedrank([min(Lhat.rf,[],2),min(Lhat.rerf,[],2),...
    min(Lhat.rf_rot,[],2),min(Lhat.rerfr,[],2),min(Lhat.frc,[],2)]')');

Classifiers = {'rf' 'rerf' 'rf_rot' 'rerfr' 'frc'};

RankCounts = NaN(5,length(Classifiers));
for i = 1:5
    RankCounts(i,:) = sum(ClRanks==i);
end

figure
barh(flipud(RankCounts))
ax = gca;
Bars = allchild(ax);
for i = 1:length(Bars)
    Bars(i).EdgeColor = 'w';
    Bars(i).BarWidth = 1;
end

xlabel('Frequency')

title('Affine')

ax.FontSize = 14;
ax.YTickLabel = {'5' '4' '3' '2' '1'}';

l = legend('RF','RerF','RR-RF','RerF(r)','F-RC');
l.Location = 'northeast';
l.Box = 'off';

save_fig(gcf,'~/Benchmarks/Classifier_ranks_affine_horizontal')