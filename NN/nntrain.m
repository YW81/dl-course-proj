function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6, 'number of input arguments must be 4 or 6')

eval.train.error           = [];
eval.train.accuracy        = [];
eval.val.error             = [];
eval.val.accuracy          = [];
eval.train.time            = []; % log the training time
update_monitor.value       = {};
update_monitor.rate        = {};
nn.eval = eval;
nn.update_monitor = update_monitor;

opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot > 0
    fhandle = figure();
end

m = size(train_x, 1); % number of training examples
batchsize = opts.batchsize;
numepochs = opts.numepochs;
numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches, 1);
n = 1;
for i = 1 : numepochs
    kk = randperm(m);
    update_value = zeros(numbatches, nn.n-1);
    update_rate = zeros(numbatches, nn.n-1);
    tic;
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :); % l(th) batch
        
        % Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x .* (rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        n = n + 1;
        update_value(l,:) = nn.update_value;
        update_rate(l,:) = nn.update_rate;
    end
    
    t = toc;
    nn.eval.train.time(end+1) = t;
    nn.update_monitor.value{end+1} = mean(update_value);
    nn.update_monitor.rate{end+1} = mean(update_rate);
    
    if opts.validation == 1
        nn = nneval(nn, train_x, train_y, val_x, val_y);
        str_perf = sprintf(', full-batch train mse = %f, val mse = %f', nn.eval.train.error(end), nn.eval.val.error(end));
    else
        nn = nneval(nn, train_x, train_y);
        str_perf = sprintf(', full-batch train mse = %f', nn.eval.train.error(end));
    end
    
    % save log
    if isfield(opts, 'savenet') && opts.savenet > 0
        if ~isfield(opts, 'savepath')
            opts.savepath = '.';
        end
        if ~isfield(opts, 'netname')
            opts.netname = 'default';
        end
        save([opts.savepath '/' opts.netname '-' mat2str(nn.size) '-' nn.activation_function '-' nn.output '.mat'], 'nn', 'opts');
    end
    
    % plot
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, opts, i);
    end
    
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ' (' num2str(t) ' sec), mini-batch train mse = ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    disp(['  update value: 10 ^ ' mat2str(log10(nn.update_monitor.value{end}))]);
    disp(['  update rate : 1e-3 ' mat2str(1e3*nn.update_monitor.rate{end})]);
    
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end
end

