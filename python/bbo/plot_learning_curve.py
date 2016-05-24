import sys
import numpy as np
import matplotlib.pyplot as plt

from bbo_plotting import plotLearningCurve, plotExplorationCurve
from bbo_plotting import loadLearningCurve, loadExplorationCurve

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2):
        print('\nUsage: '+sys.argv[0]+' <directory>\n')
        sys.exit()
        
    directory = str(sys.argv[1])
    exploration_curve = loadExplorationCurve(directory)
    learning_curve = loadLearningCurve(directory)
    
    if exploration_curve!=None: # Plot exploration too?
        fig = plt.figure(1,figsize=(16, 6))
        plotExplorationCurve(exploration_curve,fig.add_subplot(121))
        plotLearningCurve(learning_curve,fig.add_subplot(122))
    else:
        fig = plt.figure(1,figsize=(8, 6))
        plotLearningCurve(learning_curve,fig.add_subplot(111))
        
    plt.show()
