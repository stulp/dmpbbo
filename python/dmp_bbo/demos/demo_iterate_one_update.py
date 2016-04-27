import os
import sys
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../../../python')
sys.path.append(lib_path)

from demo_fake_robot import performRolloutsFakeRobot, getTaskSolver
from demo_one_update import doOneUpdate
from dmp_bbo.dmp_bbo_plotting import plotOptimizationRollouts

if __name__=="__main__":
    # See if input directory was passed
    if (len(sys.argv)==2):
        directory = str(sys.argv[1])
    else:
        print '\nUsage: '+sys.argv[0]+' <directory>\n';
        sys.exit()
  
    n_updates = 50
    for i_update in range(n_updates):
        
        # Run the update (requires no information except directory)
        doOneUpdate(directory)
        
        # Run the rollout, e.g. on your robot
        # Usually, this would not be called in a loop, but you would call
        # it manually (depending on however your robot should be called)
        update_dir = '%s/update%05d' % (directory, i_update+1)
        performRolloutsFakeRobot(update_dir)

    # Plot the optimization results (from the files saved to disk)
    fig = plt.figure(1,figsize=(15, 5))
    task_solver = getTaskSolver()
    plotOptimizationRollouts(directory,fig,task_solver)
    plt.show()

