function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    update_value = zeros(1, nn.n-1);
    update_rate = zeros(1, nn.n-1);
    
    for i = 1 : (nn.n - 1)
        if (nn.weightPenaltyL2 > 0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
        dW = nn.learningRate * dW;
        
        if (nn.momentum > 0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW; % remember
            dW = nn.vW{i};
        end
        update_value(i) = norm(dW, 'fro') / numel(dW);
        update_rate(i) = norm(dW, 'fro') / norm(nn.W{i}, 'fro');
        nn.W{i} = nn.W{i} - dW;
    end
    
    nn.update_value = update_value;
    nn.update_rate = update_rate;
end
