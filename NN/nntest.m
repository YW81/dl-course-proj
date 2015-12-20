function [er, bad] = nntest(nn, x, y)
    labels = nnpredict(nn, x); % predicted label
    [~, expected] = max(y, [], 2); % true label
    bad = find(labels ~= expected); % number of wrong classified
    er = numel(bad) / size(x, 1);
end
