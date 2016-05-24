import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from dmp_bbo_plotting import plotOptimizationRollouts

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)==2):
      directory = str(sys.argv[1])
    else:
      print('\nUsage: '+sys.argv[0]+' <directory>\n')
      sys.exit()
    
    fig = plt.figure()
    plot_rollout = None
    plotOptimizationRollouts(directory,fig,plot_rollout)
    plt.show()

