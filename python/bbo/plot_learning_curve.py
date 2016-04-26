import sys
import numpy as np
import matplotlib.pyplot as plt

from bbo_plotting import plotLearningCurve, plotExplorationCurve, plotUpdateLines

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2):
        print '\nUsage: '+sys.argv[0]+' <learning_curve_file.txt>\n';
        sys.exit()
        
    filename = str(sys.argv[1])
    learning_curve = np.loadtxt(filename)
    
    if learning_curve.shape[1]>2: # Plot exploration too?
        
        fig = plt.figure(1,figsize=(16, 6))
        ax_explo = fig.add_subplot(121)
        ax_cost = fig.add_subplot(122)

        plotExplorationCurve(learning_curve[:,0],learning_curve[:,2],ax_explo)
        plotUpdateLines(learning_curve[:,0],ax_explo)
        
        
    else:
        fig = plt.figure(1,figsize=(8, 6))
        ax_cost = fig.add_subplot(111)
        
    plotLearningCurve(learning_curve[:,0],learning_curve[:,1],ax_cost)
    plotUpdateLines(learning_curve[:,0],ax_cost)
    
    plt.show()
