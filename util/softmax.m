function p = softmax(p)
%p(p==Inf) = 1e6;
%p(p==-Inf) = 1e-6;

p = exp(bsxfun(@minus, p, max(p,[],2))); % minus its own max value
p = bsxfun(@rdivide, p, sum(p, 2)); % normalize (on each data)
end