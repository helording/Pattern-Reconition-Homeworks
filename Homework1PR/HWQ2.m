% define variables
m = 40;
n = 100;
y= zeros(1, 100);
z= zeros(1, 100);

% for each sparsity find the success ratio
for sparsity = 1:100
	% X is a matrix that exists in the Reals
	 X = randn(m, n);
	% weight. One dimensional vector so Rank of X is larger than w
	 weight = randn(n, 1);

	% if the sparsity is 100 then break
	 if (sparsity = 100)
		break
	% else randomly select non-zero components for weight according the Gaussian distribution
	else
		 nonZeroWeight = randperm(100, 100 - sparsity)
	 	 for i = 1:100 - sparsity
	  		 weight(nonZeroWeight(i)) = 0;
	 	 end
	 end
	% compression sensing result
	Y = X * weight
	% find the result solution weight
	cvx_begin
		 variable solutionWeight(n)
		 minimize( norm(solutionWeight, 1)) subject to Y == X * solutionWeight;
	cvx_end
	% save the results to be graphed later
	z(K) = norm(solutionWeight - weight, 1);
	y(K) = (norm(solutionWeight - weight, 1) < 0.01);
end
suceess_rate = sum(y(:) == 1)/100;
plot(1:100 , z)
xlabel('K')
ylabel('$\|\hat w - \bar w\|$ ',' Interpreter','latex','Fontsize', 16)
