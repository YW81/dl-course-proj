clear all; close all; clc;

%% global settings
%architecture = [784 100 10]; % an example of a shallow net
architecture = [784 1000 500 250 125 100 50 25 10]; % an example of a deep net
% use this to denote whether we use sae to do layer-wise pre-training
use_sae = 0;

%% data preprocessing
addpath(genpath('.'));
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% SAE (layer-wise pre-training)
if use_sae > 0
    % to speed up AE training, you can use the well-designed initials,
    % since shallow nets will usually converge
    opts.netname = 'dae';
    opts.savenet = 0; % no need to save
    opts.plot = 1; % only monitor the error
    opts.numepochs = 50; % number of full sweeps through data
    opts.batchsize = 100; %  mini-batch SGD
    sae = saesetup(architecture(1:end-1));
    % set paras for each auto-encoder
    for k = 1 : length(architecture)-2
        sae.ae{k}.inputZeroMaskedFraction = 0.5; % add zero-mask noise
        sae.ae{k}.activation_function = 'sigm';
        sae.ae{k}.output = 'sigm';
    end
    % train the stacked denosing autoencoders
    sae = saetrain(sae, train_x, opts);
end

%% FFNN
% parameters
opts.numepochs = 100; % number of full sweeps through data
opts.batchsize = 100; %  take a mean gradient step over this many samples
opts.savenet = 0;
opts.plot = 3; % monitot the error and accuracy
opts.netname = 'ffnn';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% you can run different activation functions
%activations = {'sigm', 'tanh', 'opttanh', 'ReLU', 'softplus'};
activations = {'sigm'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% see NN/nntrain to know more info about evals and monitors
evals = cell(1, length(activations)); % log error, accuracy and time
update_monitors = cell(1, length(activations)); % log paras change info

for k = 1 : length(activations)
    clear nn;
    nn = nnsetup(architecture);
    if use_sae > 0
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % if use_sae == 0, this net will use default initials in nnsetup  %
        % use SDAE to initialize the network                              %
        for kk = 1 : length(architecture)-2                               %
            nn.W{kk} = sae.ae{kk}.W{1};                                   %
        end                                                               %
        % through this, you can testify whether                           %
        % layer-wise pre-training provides the net with wise initials     %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    % set activation function for this loop
    nn.activation_function = activations{k};
    % for multi-class, usually use softmax to get a probability distribution
    nn.output = 'softmax';
    % a trick here, we use test set as val set to make full use of nntrain,
    % no need to call nntest here, since we just care the trend
    [nn, L] = nntrain(nn, train_x, train_y, opts, test_x, test_y);
    % save info
    evals{k} = nn.eval; % log evals for this loop
    update_monitors{k} = nn.update_monitor; % log monitors for this loop
end

%% plot final results
% plot accuracy info
colors = {'k', 'b', 'c', 'r', 'm'};
figure; hold on; legends = {};
idx = 1 : opts.numepochs;
for k = 1 : length(activations)
    plot(idx, evals{k}.train.accuracy, [colors{k} '-.'], 'LineWidth', 1.5);
    plot(idx, evals{k}.val.accuracy, [colors{k} '-'], 'LineWidth', 1.5);
    legends{end+1} = [activations{k} ' (train)'];
    legends{end+1} = [activations{k} ' (validation)'];
end
legend(legends); xlabel('Number of epochs'); ylabel('Accuracy');
title('Accuracy using different activation-funcs'); grid on;

% plot time information
tictocs = zeros(length(activations), opts.numepochs);
for k = 1 : length(activations)
    tictocs(k,:) = evals{k}.train.time;
end
figure; boxplot(tictocs', activations);
xlabel('Activation functions'); ylabel('Training time (s)')
title('Boxplot of training time'); grid on;

% plot update info of parameters
for k = 1 : length(activations)
    values = zeros(opts.numepochs, nn.n-1);
    rates = zeros(opts.numepochs, nn.n-1);
    for kk = 1 : opts.numepochs
        values(kk,:) = update_monitors{k}.value{kk};
        rates(kk,:) = update_monitors{k}.rate{kk};
    end
    % plot update-value (absolute value)
    figure; semilogy(idx', values);
    legends = {};
    for i = 1 : nn.n-1
        legends{end+1} = [num2str(i) ' -> ' num2str(i+1)];
    end
    legend(legends);
    xlabel('Number of epochs'); ylabel('Update value');
    title(['Update value of ' activations{k}]);
    % plot update-rate (relative value), should be around 1e-3
    figure; semilogy(idx', rates);
    legends = {};
    for i = 1 : nn.n-1
        legends{end+1} = [num2str(i) ' -> ' num2str(i+1)];
    end
    legend(legends);
    xlabel('Number of epochs'); ylabel('Update rate');
    title(['Update rate of ' activations{k}]);
end

%% end of script