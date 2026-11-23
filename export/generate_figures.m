%% generate_figures.m
% generates all classifier result figures as .fig files with Sixers color palette.
%
% prerequisites:
%   - run export_to_mat.py to generate classifier_results.mat
%
% usage:
%   generate_figures          % loads from models/classifier_results.mat
%   generate_figures('path/to/results.mat')
%
% output:
%   all .fig files saved to the same directory as the .mat file.
%
% author: Michael Hilt
% date: 2026-07-19
function generate_figures(mat_file)
    if nargin < 1
        % Default: models/classifier_results.mat at the repository root
        % (this script lives in export/, so the root is one level up).
        script_dir = fileparts(mfilename('fullpath'));
        repo_root = fileparts(script_dir);
        mat_file = fullfile(repo_root, 'models', 'classifier_results.mat');
    end
    if ~isfile(mat_file)
        error(['Results file not found: %s\n' ...
               'Run: python export/export_to_mat.py'], mat_file);
    end
    % load data
    data = load(mat_file);
    [output_dir, ~, ~] = fileparts(mat_file);
    if isempty(output_dir)
        output_dir = '.';
    end
    fprintf('Loaded results from: %s\n', mat_file);
    fprintf('Output directory: %s\n\n', output_dir);
    %% Sixers color palette (in listed order)
    sixersColors = [
        0,   107, 182;  % 1  blue (#006bb6)
        237, 23,  76;   % 2  red (#ed174c)
        0,   43,  92;   % 3  navy (#002b5c)
        0,   122, 51;   % 4  green (#007a33)
        0,   169, 224;  % 5  light blue (#00a9e0)
        150, 56,  33;   % 6  brown (#963821)
        187, 151, 84;   % 7  gold (#bb9754)
        45,  45,  45;   % 8  charcoal
        245, 180, 0;    % 9  orange-yellow
        50,  75,  30;   % 10 dark olive
        250, 190, 210;  % 11 light pink
        120, 20,  120;  % 12 dark plum
        255, 245, 180;  % 13 light yellow
        0,   150, 150;  % 14 medium teal
        255, 160, 122   % 15 light coral
    ] / 255;
    % class colors (blue, red, navy)
    classColors = sixersColors([1, 2, 3], :);
    %% determine which models were trained
    models = cellstr(data.models_trained);
    modelDisplay = containers.Map({'knn','svm','rf'}, {'KNN','SVM','Random Forest'});
    %% class distribution (pie charts)
    fprintf('Generating: class_distribution.fig\n');
    fig1 = figure('Name', 'Class Distribution', 'Position', [100 100 1000 450], 'Visible', 'off');
    classLabels = cellstr(data.class_labels);
    % training set
    subplot(1, 2, 1);
    trainCounts = data.train_class_counts(:);
    pie(trainCounts);
    colormap(gca, classColors);
    title(sprintf('Training Set (n=%d)', data.training_samples), 'FontSize', 11);
    legend(classLabels, 'Location', 'southoutside', 'FontSize', 9);
    % test set
    subplot(1, 2, 2);
    testCounts = data.test_class_counts(:);
    pie(testCounts);
    colormap(gca, classColors);
    title(sprintf('Test Set (n=%d)', data.test_samples), 'FontSize', 11);
    legend(classLabels, 'Location', 'southoutside', 'FontSize', 9);
    sgtitle('Class Distribution', 'FontSize', 13);
    savefig(fig1, fullfile(output_dir, 'class_distribution.fig'));
    close(fig1);
    %% confusion matrices
    for k = 1:numel(models)
        m = models{k};
        cmField = [m, '_confusion_matrix'];
        if ~isfield(data, cmField)
            continue;
        end
        displayName = modelDisplay(m);
        fprintf('Generating: %s_confusion_matrix.fig\n', m);
        cm = data.(cmField);
        cmNorm = cm ./ sum(cm, 2);  % normalize rows
        fig = figure('Name', [displayName ' Confusion Matrix'], ...
                     'Position', [100 100 700 550], 'Visible', 'off');
        imagesc(cmNorm, [0 1]);
        colormap(gca, blues_colormap());
        cb = colorbar;
        cb.Label.String = 'Proportion';
        ax = gca;
        ax.XTick = 1:3;
        ax.YTick = 1:3;
        ax.XTickLabel = classLabels;
        ax.YTickLabel = classLabels;
        ax.XTickLabelRotation = 30;
        xlabel('Predicted Label', 'FontSize', 11);
        ylabel('True Label', 'FontSize', 11);
        title(sprintf('%s — Confusion Matrix', displayName), 'FontSize', 12);
        % annotate cells
        for i = 1:3
            for j = 1:3
                pct = cmNorm(i, j) * 100;
                txt = sprintf('%d\n(%.1f%%)', cm(i, j), pct);
                if cmNorm(i, j) > 0.5
                    txtColor = 'w';
                else
                    txtColor = 'k';
                end
                text(j, i, txt, 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', 'FontSize', 10, ...
                     'Color', txtColor);
            end
        end
        figName = sprintf('%s_confusion_matrix.fig', m);
        savefig(fig, fullfile(output_dir, figName));
        close(fig);
    end
    %% model comparison bar chart
    if numel(models) > 1
        fprintf('Generating: model_comparison.fig\n');
        fig3 = figure('Name', 'Model Comparison', 'Position', [100 100 900 450], 'Visible', 'off');
        metricNames = {'accuracy', 'precision', 'recall', 'f1'};
        metricLabels = {'Accuracy', 'Precision', 'Recall', 'F1 Score'};
        nMetrics = numel(metricNames);
        nModels = numel(models);
        barData = zeros(nMetrics, nModels);
        for k = 1:nModels
            for j = 1:nMetrics
                fieldName = [models{k}, '_', metricNames{j}];
                barData(j, k) = data.(fieldName);
            end
        end
        barColors = sixersColors([1, 2, 4], :);  % Blue, Red, Green
        x = 1:nMetrics;
        width = 0.8 / nModels;
        hold on;
        for k = 1:nModels
            offset = (k - (nModels+1)/2) * width;
            b = bar(x + offset, barData(:, k), width, 'FaceColor', barColors(k, :), ...
                    'EdgeColor', 'w', 'LineWidth', 0.5);
            % Value labels
            for j = 1:nMetrics
                text(x(j) + offset, barData(j, k) + 0.002, ...
                     sprintf('%.3f', barData(j, k)), ...
                     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                     'FontSize', 8);
            end
        end
        hold off;
        set(gca, 'XTick', x, 'XTickLabel', metricLabels);
        ylim([0.9, 1.01]);
        ylabel('Score');
        title('Model Comparison — Classification Performance', 'FontSize', 12);
        legend(cellfun(@(m) modelDisplay(m), models, 'UniformOutput', false), ...
               'Location', 'southwest');
        grid on;
        set(gca, 'GridAlpha', 0.3, 'YGrid', 'on', 'XGrid', 'off');
        savefig(fig3, fullfile(output_dir, 'model_comparison.fig'));
        close(fig3);
    end
    %% per-class Metrics
    if numel(models) >= 1
        fprintf('Generating: per_class_metrics.fig\n');
        fig4 = figure('Name', 'Per-Class Metrics', ...
                      'Position', [100 100 500*numel(models) 450], 'Visible', 'off');
        for k = 1:numel(models)
            m = models{k};
            cmField = [m, '_confusion_matrix'];
            if ~isfield(data, cmField)
                continue;
            end
            cm = data.(cmField);
            nClasses = size(cm, 1);
            precisions = zeros(nClasses, 1);
            recalls = zeros(nClasses, 1);
            f1s = zeros(nClasses, 1);
            for c = 1:nClasses
                tp = cm(c, c);
                fp = sum(cm(:, c)) - tp;
                fn = sum(cm(c, :)) - tp;
                p = tp / max(tp + fp, 1);
                r = tp / max(tp + fn, 1);
                f = 2*p*r / max(p + r, eps);
                precisions(c) = p;
                recalls(c) = r;
                f1s(c) = f;
            end
            subplot(1, numel(models), k);
            x = 1:nClasses;
            w = 0.25;
            hold on;
            bar(x - w, precisions, w, 'FaceColor', sixersColors(1, :));  % Blue
            bar(x,     recalls,    w, 'FaceColor', sixersColors(4, :));  % Green
            bar(x + w, f1s,        w, 'FaceColor', sixersColors(2, :));  % Red
            hold off;
            set(gca, 'XTick', x, 'XTickLabel', arrayfun(@(c) sprintf('C%d', c-1), x, 'Uni', false));
            ylim([0.85, 1.02]);
            ylabel('Score');
            title(modelDisplay(m), 'FontSize', 11);
            legend({'Precision', 'Recall', 'F1'}, 'FontSize', 8, 'Location', 'southwest');
            grid on;
            set(gca, 'GridAlpha', 0.3, 'YGrid', 'on', 'XGrid', 'off');
        end
        sgtitle('Per-Class Performance Metrics', 'FontSize', 13);
        savefig(fig4, fullfile(output_dir, 'per_class_metrics.fig'));
        close(fig4);
    end
    %% RF Feature Importance — Top 30
    if isfield(data, 'rf_top30_importances')
        fprintf('Generating: random_forest_feature_importance.fig\n');
        fig5 = figure('Name', 'RF Feature Importance', ...
                      'Position', [100 100 900 850], 'Visible', 'off');
        topNames = cellstr(data.rf_top30_names);
        topImps = data.rf_top30_importances(:);
        nTop = numel(topImps);
        % assign colors by feature type
        typeColorMap = struct( ...
            'S11_mag_dB',    sixersColors(1, :), ...
            'S11_phase',     sixersColors(3, :), ...
            'S11_grpdelay',  sixersColors(5, :), ...
            'S11_mag_slope', sixersColors(14, :), ...
            'S21_mag_dB',    sixersColors(2, :), ...
            'S21_phase',     sixersColors(6, :), ...
            'S21_grpdelay',  sixersColors(9, :), ...
            'S21_mag_slope', sixersColors(7, :), ...
            'stat',          sixersColors(12, :));
        barColors = zeros(nTop, 3);
        for i = 1:nTop
            name = topNames{i};
            if startsWith(name, 'S11_mag_dB')
                barColors(i, :) = typeColorMap.S11_mag_dB;
            elseif startsWith(name, 'S11_phase')
                barColors(i, :) = typeColorMap.S11_phase;
            elseif startsWith(name, 'S11_grpdelay')
                barColors(i, :) = typeColorMap.S11_grpdelay;
            elseif startsWith(name, 'S11_mag_slope')
                barColors(i, :) = typeColorMap.S11_mag_slope;
            elseif startsWith(name, 'S21_mag_dB')
                barColors(i, :) = typeColorMap.S21_mag_dB;
            elseif startsWith(name, 'S21_phase')
                barColors(i, :) = typeColorMap.S21_phase;
            elseif startsWith(name, 'S21_grpdelay')
                barColors(i, :) = typeColorMap.S21_grpdelay;
            elseif startsWith(name, 'S21_mag_slope')
                barColors(i, :) = typeColorMap.S21_mag_slope;
            else
                barColors(i, :) = typeColorMap.stat;
            end
        end
        % horizontal bar chart (top at top)
        y = nTop:-1:1;
        hold on;
        for i = 1:nTop
            barh(y(i), topImps(i), 'FaceColor', barColors(i, :), 'EdgeColor', 'none');
        end
        hold off;
        
        set(gca, 'YTick', sort(y), 'YTickLabel', topNames(end:-1:1), 'FontSize', 8);
        
        xlabel('Feature Importance');
        title('Random Forest — Top 30 Feature Importances', 'FontSize', 12);
        ylim([0.5, nTop + 0.5]);
        % legend
        legendLabels = {'S11 Magnitude', 'S11 Phase', 'S11 Group Delay', 'S11 Mag Slope', ...
                        'S21 Magnitude', 'S21 Phase', 'S21 Group Delay', 'S21 Mag Slope', ...
                        'Statistical'};
        legendColors = [typeColorMap.S11_mag_dB; typeColorMap.S11_phase; ...
                        typeColorMap.S11_grpdelay; typeColorMap.S11_mag_slope; ...
                        typeColorMap.S21_mag_dB; typeColorMap.S21_phase; ...
                        typeColorMap.S21_grpdelay; typeColorMap.S21_mag_slope; ...
                        typeColorMap.stat];
        hold on;
        hLeg = gobjects(numel(legendLabels), 1);
        for i = 1:numel(legendLabels)
            hLeg(i) = patch(NaN, NaN, legendColors(i, :), 'EdgeColor', 'none');
        end
        hold off;
        legend(hLeg, legendLabels, 'Location', 'southeast', 'FontSize', 8);
        savefig(fig5, fullfile(output_dir, 'random_forest_feature_importance.fig'));
        close(fig5);
    end
    %% RF Feature Importance by Frequency
    if isfield(data, 'rf_imp_s11_mag')
        fprintf('Generating: random_forest_importance_by_freq.fig\n');
        fig6 = figure('Name', 'RF Importance by Frequency', ...
                      'Position', [100 100 1200 900], 'Visible', 'off');
        freqGHz = data.freq_ghz(:);
        freqMid = (freqGHz(1:end-1) + freqGHz(2:end)) / 2;
        % feature groups: {data, freq_axis, title, color_index}
        groups = {
            data.rf_imp_s11_mag,      freqGHz, 'S11 Magnitude (dB)',      1;
            data.rf_imp_s11_phase,    freqGHz, 'S11 Phase (deg)',         3;
            data.rf_imp_s21_mag,      freqGHz, 'S21 Magnitude (dB)',      2;
            data.rf_imp_s21_phase,    freqGHz, 'S21 Phase (deg)',         6;
            data.rf_imp_s11_grpdelay, freqMid, 'S11 Group Delay (ns)',    5;
            data.rf_imp_s21_grpdelay, freqMid, 'S21 Group Delay (ns)',    9;
            data.rf_imp_s11_slope,    freqMid, 'S11 Mag Slope (dB/GHz)', 14;
            data.rf_imp_s21_slope,    freqMid, 'S21 Mag Slope (dB/GHz)',  7;
        };
        for i = 1:size(groups, 1)
            subplot(4, 2, i);
            imp = groups{i, 1}(:);
            fx = groups{i, 2}(:);
            ttl = groups{i, 3};
            cidx = groups{i, 4};
            % Match lengths
            n = min(numel(imp), numel(fx));
            fx = fx(1:n);
            imp = imp(1:n);
            fill([fx; flipud(fx)], [imp; zeros(n, 1)], sixersColors(cidx, :), ...
                 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            hold on;
            plot(fx, imp, 'Color', sixersColors(cidx, :), 'LineWidth', 0.8);
            hold off;
            ylabel('Importance', 'FontSize', 8);
            title(ttl, 'FontSize', 10);
            grid on;
            set(gca, 'GridAlpha', 0.3);
            xlim([freqGHz(1), freqGHz(end)]);
            if i >= 7
                xlabel('Frequency (GHz)');
            end
        end
        sgtitle('Random Forest — Feature Importance by Frequency', 'FontSize', 14);
        savefig(fig6, fullfile(output_dir, 'random_forest_importance_by_freq.fig'));
        close(fig6);
    end
    %% done
    fprintf('\nAll .fig files saved to: %s/\n', output_dir);
    fprintf('Open in MATLAB with: openfig(''filename.fig'')\n\n');
end
%% helper: generate a blue colormap similar to matplotlib Blues
function cmap = blues_colormap()
    n = 256;
    % From white to Sixers blue (#006bb6 = [0, 107, 182]/255)
    r = linspace(1, 0/255, n)';
    g = linspace(1, 107/255, n)';
    b = linspace(1, 182/255, n)';
    cmap = [r, g, b];
end