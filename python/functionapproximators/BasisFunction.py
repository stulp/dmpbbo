# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import numpy.matlib

class Gaussian:
    def activations(centers, widths, inputs, normalized_basis_functions=False):

        assert(centers.shape==widths.shape)

        n_basis_functions = centers.size
        n_samples         = inputs.size
        n_dims            = 1
  
        kernel_activations = np.ones([n_samples,n_basis_functions])
  
        if normalized_basis_functions and n_basis_functions==1:
            # Locally Weighted Regression with only one basis function is pretty odd.
            # Essentially, you are taking the "Locally Weighted" part out of the regression, and it becomes
            # standard least squares 
            # Anyhow, for those that still want to "abuse" LWR as R (i.e. without LW), we explicitly
            # set the normalized kernels to 1 here, to avoid numerical issues in the normalization below.
            # (normalizing a Gaussian basis function with itself leads to 1 everywhere).
            kernel_activations.fill(1.0)
            return kernel_activations
  
  
        for bb in range(n_basis_functions):
            # Here, we compute the values of a (unnormalized) multi-variate Gaussian:
            #   activation = exp(-0.5*(x-mu)*Sigma^-1*(x-mu))
            # Because Sigma is diagonal in our case, this simplifies to
            #   activation = exp(\sum_d=1^D [-0.5*(x_d-mu_d)^2/Sigma_(d,d)]) 
            #              = \prod_d=1^D exp(-0.5*(x_d-mu_d)^2/Sigma_(d,d)) 
            # This last product is what we compute below incrementally
    

            for i_dim in range(n_dims):
               c = centers[bb] # c = centers[bb,i_dim]
               for i_s in range(n_samples):
                   x = inputs[i_s] # x = inputs[i_s,i_dim]
                   w = widths[bb]  # w = widths[bb,i_dim]
        
                   kernel_activations[i_s,bb] *= np.exp(-0.5*np.square(x-c)/(w*w))
                   
        #np.set_printoptions(precision=3,suppress=True)
        #print(kernel_activations)
                   
        if (normalized_basis_functions):
            # Normalize the basis value; they should sum to 1.0 for each time step.
            for i_sample in range(n_samples):
                sum_kernel_activations = kernel_activations[i_sample,:].sum()
                for i_basis in range(n_basis_functions):
                    if (sum_kernel_activations==0.0):
                        # Apparently, no basis function was active. Set all to same value
                        kernel_activations[i_sample,i_basis] = 1.0/n_basis_functions
                    else:
                        # Standard case, normalize so that they sum to 1.0
                        kernel_activations[i_sample,i_basis] /= sum_kernel_activations
                        
        return kernel_activations