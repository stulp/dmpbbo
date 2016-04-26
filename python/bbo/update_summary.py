import os
import numpy as np

from distribution_gaussian import DistributionGaussian

class UpdateSummary:
    
    def __init__(self,distributions,samples,cost_eval,costs,weights,distributions_new,cost_vars=None,cost_vars_eval=None):

        if isinstance(distributions,DistributionGaussian):
            # distributions should be a list of DistributionGaussian
            self.distributions = [distributions]
        else:
            self.distributions = distributions
            
        if isinstance(distributions_new,DistributionGaussian):
            # distributions should be a list of DistributionGaussian
            self.distributions_new = [distributions_new]
        else:
            self.distributions_new = distributions_new
        
        self.samples = samples
        self.cost_eval = cost_eval
        self.costs = costs
        self.weights = weights
        self.cost_vars = cost_vars
        self.cost_vars_eval = cost_vars_eval
        
            
    def saveToDirectory(self,directory,overwrite=False):
        """ Save an update summary to a directory
        \param[in] directory Directory to which to write object
        \param[in] overwrite Overwrite existing files in the directory above (default: false)
        """
        
        # Make directory if it doesn't already exist
        if not os.path.isdir(directory): # ZZZ overwrite?
            os.makedirs(directory) 
        
        d = directory+'/'
        np.savetxt(d+'distribution_mean.txt',     self.distribution.mean)
        np.savetxt(d+'distribution_covar.txt',    self.distribution.covar)
        np.savetxt(d+'samples.txt',               self.samples)
        np.savetxt(d+'costs.txt',                 self.costs)
        np.savetxt(d+'weights.txt',               self.weights)
        np.savetxt(d+'distribution_new_mean.txt', self.distribution_new.mean)
        np.savetxt(d+'distribution_new_covar.txt',self.distribution_new.covar)
        
        if self.cost_eval:
            print(self.cost_eval)
            number_to_save = np.atleast_1d(self.cost_eval   )
            print(number_to_save)
            np.savetxt(d+'cost_eval.txt',    number_to_save)
        if self.cost_vars_eval != None:
            np.savetxt(d+'cost_vars_eval.txt',self.cost_vars_eval)
        if self.cost_vars != None:
            np.savetxt(d+'cost_vars.txt',self.cost_vars)

def loadFromDirectory(directory):
    
    # Read data
    try:
        n_parallel = np.loadtxt(directory+"/n_parallel.txt")
    except IOError:
        n_parallel = 1
    
    if n_parallel==1:
        mean = np.loadtxt(directory+"/distribution_mean.txt")
        covar = np.loadtxt(directory+"/distribution_covar.txt")
        distributions = [DistributionGaussian(mean,covar)]
        
        mean = np.loadtxt(directory+"/distribution_new_mean.txt")
        covar = np.loadtxt(directory+"/distribution_new_covar.txt")
        distributions_new = [DistributionGaussian(mean,covar)]

    else:
        distributions = []
        distributions_new = []
        for i_parallel in range(n_parallel):
            cur_file = '%s/distribution_%03d' % (directory, i_parallel)
            mean = np.loadtxt(cur_file+'_mean.txt')
            covar = np.loadtxt(cur_file+'_covar.txt')
            distributions.append(DistributionGaussian(mean,covar))
            
            cur_file = '%s/distribution_new_%03d' % (directory, i_parallel)
            mean = np.loadtxt(cur_file+'_mean.txt')
            covar = np.loadtxt(cur_file+'_covar.txt')
            distributions_new.append(DistributionGaussian(mean,covar))

    
    try:
        samples = np.loadtxt(directory+"/samples.txt")
    except IOError:
        samples = None
    costs = np.loadtxt(directory+"/costs.txt")
    weights = np.loadtxt(directory+"/weights.txt")

    
    try:
        cost_eval = np.loadtxt(directory+"/cost_eval.txt")
    except IOError:
        cost_eval = None

    n_rollouts = len(weights)
    cost_vars = []
    for i_rollout in range(n_rollouts):
        cur_directory = '%s/rollout%05d' % (directory, i_rollout)
        try:
            cost_vars.append(np.loadtxt(cur_directory+"/cost_vars.txt"))
        except IOError:
            pass

    try:
        cost_vars_eval = np.loadtxt(directory+"/rollout_eval/cost_vars.txt")
    except IOError:
        cost_vars_eval = None

    
    return UpdateSummary(distributions,samples,cost_eval,costs,weights,distributions_new,cost_vars,cost_vars_eval)


def extractLearningCurve(update_summaries):
    
    # Memory for the learning curve
    n_updates = len(update_summaries)
    learning_curve = np.full((n_updates,4),0.0)
    learning_curve[0,0] = 0 # First evaluation is at 0

    
    for i_update in range(n_updates):
      
        cur_summary = update_summaries[i_update]

        # Number of samples at which an evaluation was performed.
        if (i_update>0):
            n_samples = len(update_summaries[i_update].costs)
            learning_curve[i_update,0] = learning_curve[i_update-1,0] + n_samples 
    
        # The cost of the evaluation at this update
        learning_curve[i_update,1] = cur_summary.cost_eval
    
        # The largest eigenvalue of the covariance matrix
        learning_curve[i_update,2] = 0
        for distribution in cur_summary.distributions:
            eigen_values = np.linalg.eigvals(distribution.covar)
            learning_curve[i_update,2] += np.sqrt(max(eigen_values))
    
        #mean_costs = np.mean(cur_summary.costs)
        std_costs = np.std(cur_summary.costs)
        learning_curve[i_update,3] = std_costs;
    
    return learning_curve

    
def saveToDirectory(update_summaries, directory, overwrite=False, only_learning_curve=False):
    """ Save a vector of update summaries to a directory
    \param[in] directory Directory to which to write object
    \param[in] overwrite Overwrite existing files in the directory above (default: false)
    \param[in] only_learning_curve Save only the learning curve (default: false)
    """
    curves = extractLearningCurve(update_summaries)
    np.savetxt(directory+'/learning_curve.txt',self.curves)

    if only_learning_curve:
        return
  
    for i_update in range(n_updates):
      
        cur_summary = update_summaries[i_update]
        # Save all the information in the update summaries
        cur_directory = '%s/update%05d' % (directory, i_update+1)
        cur_summary.saveToDirectory(cur_directory,overwrite)

