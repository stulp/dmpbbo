# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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
import math
import os

class DistributionGaussian:

    def __init__(self, mean=0.0, covar=1.0):
        self.mean = mean
        self.covar = covar

    def generateSamples(self,n_samples=1):
        return np.random.multivariate_normal(self.mean, self.covar, n_samples)

    def maxEigenValue(self):
        return max(np.linalg.eigvals(self.covar))

    def __str__(self):
        return 'N( '+str(self.mean)+', '+str(self.covar)+' )'

    def saveToDirectory(self,directory,basename):
        if not os.path.exists(directory):
              os.makedirs(directory)
        d = directory
        np.savetxt(directory+'/'+basename+'_mean.txt',self.mean)
        np.savetxt(directory+'/'+basename+'_covar.txt',self.covar)

def loadDistributionGaussianFromDirectory(directory,basename):
    mean = np.loadtxt(directory+'/'+basename+'_mean.txt')
    covar = np.loadtxt(directory+'/'+basename+'_covar.txt')
    return DistributionGaussian(mean,covar)
    
#def loadDistributionGaussianFromDirectory(directory):
#    rollouts = []
#    if i_rollout:
#        cur_dir = '%s/rollout%03d' % (directory, i_rollout+1)
#        if not os.path.exists(cur_dir):
#            return rollouts
#        else:
#            rollouts.append(loadRolloutFromDirectory(cur_dir))


def testDistributionGaussian():
    mu  = np.array([2,4])
    cov = np.array([[0.1,0.3],[0.3,0.5]])
    
    d = DistributionGaussian(mu,cov)

    xs = d.generateSamples(50)
    print(xs)
    
    import matplotlib.pyplot as plt
    plt.plot(xs[:,0], xs[:,1], 'x')
    plt.axis('equal')
    plt.show()
    
if __name__ == '__main__':
    testDistributionGaussian()

