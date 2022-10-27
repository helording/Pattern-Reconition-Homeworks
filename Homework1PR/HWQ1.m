%define variables
m = 40;
n = 100;
recovery_accuracy = zeros(1, 100);

for sparsity = 1:100
    X = randn(m, n);
    weight = randn(n, 1);
    if (sparsity ~= 100)
        rand = randperm(100, 100 - sparsity)
        for i = 1:100-sparsity
            weight(rand(i)) = 0;
        end
    end

    % put noise into Y
    Y = awgn(X*weight, 20);
    weight = sparse(weight);

    lowest_err = 99999;
    best_lambda = 0;

    for lambda = 1:0.1:12
            % train set for cross validation
            train = X
            train(1:4, :) = []

            % test set
            test = X(1:4, :)

            cvx_begin
                variable solWeight(n)
                minimize(norm(solWeight, 1) + (0.5 * lambda) * (square_pos(norm(Y - train * solWeight))))
            cvx_end

            err = 0.25 * norm(test_set(:,:) * solWeight - Y)
            if err < lowest_err
              lowest_err = err
              best_lambda = lambda
    end

    cvx_begin
        variable solWeight(n)
        minimize(norm(solWeight,1) + (0.5 * best_lambda) * (square_pos(norm(Y - X * solWeight))))
    cvx_end
    recovery_accuracy(sparsity) = norm(solWeight - weight) / norm(weight)
end
plot(1:100, recovery_accuracy(1:100))
xlabel('sparsity')
ylabel('recovery accuracy')
