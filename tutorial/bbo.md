Black Box Optimization
===============

This page explains the implementation of [evolution strategies](http://en.wikipedia.org/wiki/Evolution_strategy) for the [optimization](http://en.wikipedia.org/wiki/Optimization_%28mathematics%29) of black-box [cost functions](http://en.wikipedia.org/wiki/Loss_function).

Black-box in this context means that no assumptions about the cost function can be made, for example, we do not have access to its derivative, and we do not even know if it is continuous or not.

The evolution strategies that are implemented are all based on reward-weighted averaging (aka probablity-weighted averaging), as explained in this paper/presentation: [http://icml.cc/discuss/2012/171.html](http://icml.cc/discuss/2012/171.html)

Here is a 1-page description of the algorithm:
http://www.pyoudeyer.com/stulpOudeyerDevelopmentalScience17.pdf#page=30

The basic algorithm is as follows:

	x_mu = ??; x_Sigma = ?? // Initialize multi-variate Gaussian distribution
	while (!halt_condition) {
		// Explore
		for k=1:K {
			x[k] ~ N(x_mu,x_Sigma) // Sample from Gaussian
			costs[k] = costfunction(x[k]) // Evaluate sample
		}
		// Update distribution
		weights = costs2weights(costs) // Should assign higher weights to lower costs
		x_mu_new = weightsT * x; // Compute weighted mean of samples
		x_covar_new = (weights .* x)T * weights // Compute weighted covariance matrix of samples
		x_mu = x_mu_new
		x_covar = x_covar_new
	}
	
Further reading
---------------

The next tutorials to go to would be:
* <a href="bbo_for_dmps.md">Black-Box Optimizaton for Dynamical Movement Primitives</a> (if you already know about dynamical movement primitives)
* <a href="dmp.md">Dynamical Movement Primitives</a> (if you don't)


