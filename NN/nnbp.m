function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights

n = nn.n;
sparsityError = 0;
switch nn.output
    case 'sigm'
        d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
    case {'softmax', 'linear'}
        d{n} = - nn.e;
end
%assert(any(any(isnan(d{n})))==0, 'd{n}');

for i = (n - 1) : -1 : 2 % back-propagation
    % Derivative of the activation function
    switch nn.activation_function
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i});
        case 'opttanh'
            d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
        case 'tanh'
            d_act = 1 - nn.a{i}.^2;
        case 'ReLU'
            d_act = nn.a{i} > 0;
        case 'softplus'
            d_act = (exp(nn.a{i})-1) ./ exp(nn.a{i});
    end
%assert(any(any(isnan(d_act)))==0, 'd_act');
    
    if(nn.nonSparsityPenalty>0)
        pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
        sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
    end
    
    % Backpropagate first derivatives
    if i+1==n % in this case in d{n} there is not the bias term to be removed
        d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
%assert(any(any(isnan(d{i})))==0, 'd{i}');
    else % in this case in d{i} the bias term has to be removed
        d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
%assert(any(any(isnan(d{i})))==0, 'd{i}');
    end
    
    if(nn.dropoutFraction>0)
        d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
    end
end

for i = 1 : (n - 1) % dW by this mini-batch
    if i+1==n
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1); % average over # of examples
    else
        nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);
    end
%assert(any(any(isnan(nn.dW{i})))==0, 'nn.dW{i}');
end
end
