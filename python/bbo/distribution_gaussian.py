import numpy as np
import math

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

