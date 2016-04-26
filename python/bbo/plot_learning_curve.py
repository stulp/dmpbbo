import sys
import numpy as np
import matplotlib.pyplot as plt

from bbo_plotting import plotCurve

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2):
        print '\nUsage: '+sys.argv[0]+' <learning_curve_file.txt>\n';
        sys.exit()
        
    filename = str(sys.argv[1])
    learning_curve = np.loadtxt(filename)
    
    if learning_curve.shape[1]>2: # Plot exploration too?
        fig = plt.figure(1,figsize=(16, 6))
        axs = [fig.add_subplot(121), fig.add_subplot(122)]
    else:
        fig = plt.figure(1,figsize=(8, 6))
        axs = [fig.add_subplot(111)]
        
    plotCurve(learning_curve,axs)
        
    plt.show()
