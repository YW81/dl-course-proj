function nn = nneval(nn, train_x, train_y, val_x, val_y)
%NNEVAL evaluates performance of neural network
% update nn.eval struct
assert(nargin == 3 || nargin == 5, 'Wrong number of arguments');

% calc error
nn.testing = 1;
nn = nnff(nn, train_x, train_y);
nn.eval.train.error(end+1) = nn.L;

if nargin == 5
    nn = nnff(nn, val_x, val_y);
    nn.eval.val.error(end+1) = nn.L;
end

% calc accracy
nn.testing = 0;
[er_train, ~] = nntest(nn, train_x, train_y);
nn.eval.train.accuracy(end+1) = 1 - er_train;

if nargin == 5
    [er_val, ~]  = nntest(nn, val_x, val_y);
    nn.eval.val.accuracy(end+1) = 1 - er_val;
end

end
