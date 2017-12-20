from mpl_toolkits.mplot3d import Axes3D
import numpy                                     
import math
import matplotlib.pyplot as plt                                               
import sys

# 
def plotData(inputs,outputs,ax):
    """Plot outputs against inputs, whilst being smart about the dimensionality of inputs."""
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        return ax.plot(inputs,outputs,'.')
    elif (n_dims==2):
        return ax.plot(inputs[:,0],inputs[:,1],outputs,'.')
    else:
        print('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
        return []


def plotDataTargets(inputs,target,ax):
    """Plot outputs against targets, and apply default style for targets."""
    list_of_lines = plotData(inputs,target,ax)
    plt.setp(list_of_lines, label='targets', color='black',markersize=7)                  
    return list_of_lines


def plotDataPredictions(inputs,predictions,ax):
    """Plot outputs against targets, and apply default style for predictions."""
    list_of_lines = plotData(inputs,predictions,ax)
    plt.setp(list_of_lines, label='predictions', color='red')
    return list_of_lines

def plotDataResiduals(inputs,targets,predictions,ax):

    n_dims = len(numpy.atleast_1d(inputs[0]))
        
    list_of_lines = []
    if (n_dims==1):
        for ii in range(len(inputs)):
            l = ax.plot([inputs[ii], inputs[ii]],[targets[ii], predictions[ii] ])
            list_of_lines.append(l)
    elif (n_dims==2):
        for ii in range(len(inputs)):
            l = ax.plot([inputs[ii,0], inputs[ii,0]],[inputs[ii,1], inputs[ii,1]],[targets[ii], predictions[ii] ])
            list_of_lines.append(l)
    else:
        print('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')

    plt.setp(list_of_lines, label='predictions', color='red', linewidth=2)
    return list_of_lines

def plotGrid(inputs,outputs,ax,n_samples_per_dim):
    """Plot outputs against inputs, whilst being smart about the dimensionality of inputs."""
    list_of_lines = []
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        list_of_lines = ax.plot(inputs,outputs,'-')
    elif (n_dims==2):
        inputs_0_on_grid = numpy.reshape(inputs[:,0],n_samples_per_dim)
        inputs_1_on_grid = numpy.reshape(inputs[:,1],n_samples_per_dim)
        outputs_on_grid = numpy.reshape(outputs,n_samples_per_dim)
        list_of_lines = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,outputs_on_grid,rstride=1, cstride=1)
    else:
        print('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
        
    return list_of_lines
    
def plotGridPredictions(inputs,predictions,ax,n_samples_per_dim):
    """Plot outputs against targets, and apply default style for predictions."""
    list_of_lines = plotGrid(inputs,predictions,ax,n_samples_per_dim)
    n_dims = len(numpy.atleast_1d(n_samples_per_dim))
    if (n_dims==1):
        plt.setp(list_of_lines, label='latent function', color='#9999ff',linewidth=3)
    else:
        plt.setp(list_of_lines, label='latent function', color='#5555ff',linewidth=1)
    return list_of_lines
    
def plotGridVariance(inputs,predictions,variances,ax,n_samples_per_dim=None):
    list_of_lines1 = plotGrid(inputs,predictions+2*(numpy.sqrt(variances)),ax,n_samples_per_dim)
    list_of_lines2 = plotGrid(inputs,predictions-2*(numpy.sqrt(variances)),ax,n_samples_per_dim)
    list_of_lines = [list_of_lines1, list_of_lines2]
    plt.setp(list_of_lines, linewidth=0.5, label='variance', color='#aaaaff')
    return list_of_lines
    
    
def getDataDimFromDirectory(directory):
    try:
      inputs   = numpy.loadtxt(directory+'/inputs.txt')    
    except IOError:
      inputs   = numpy.loadtxt(directory+'/inputs_grid.txt')    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    return n_dims

def plotDataFromDirectory(directory,ax):
    """Read inputs, targets and predictions from file, and plot them."""
    
    # Read data
    inputs   = numpy.loadtxt(directory+'/inputs.txt')    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims>2):
        sys.exit('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
    targets     = numpy.loadtxt(directory+'/targets.txt')                            
    try:
      predictions = numpy.loadtxt(directory+'/outputs.txt')
    except IOError:
      # Everything's fine: user did not store outputs
      predictions = [];
        
    try:
        n_samples_per_dim = numpy.loadtxt(directory+'/n_samples_per_dim.txt',dtype=int)                             
    except IOError:
        n_samples_per_dim = None
      
    # Plotting
    try:
      inputs_grid = numpy.loadtxt(directory+'/inputs_grid.txt')
      predictions_grid = numpy.loadtxt(directory+'/outputs_grid.txt')
      plotGridPredictions(inputs_grid,predictions_grid,ax,n_samples_per_dim)   
      try:
        variances_grid = numpy.loadtxt(directory+'/variances_grid.txt')
        plotGridVariance(inputs_grid,predictions_grid,variances_grid,ax,n_samples_per_dim)
      except IOError:
        # Everything's fine: user did not store grid predictions
        variances_grid = [];
    except IOError:
      # Everything's fine: user did not store grid predictions
      predictions_grid = [];
      
      
    if len(predictions)>0:
      plotDataResiduals(inputs,targets,predictions,ax)
    plotDataTargets(inputs,targets,ax)
    #plotDataPredictions(inputs,predictions,ax)   
    
    
    
    # Annotation
    if (n_dims==1):
      ax.set_xlabel('input');
      ax.set_ylabel('output');
    else:
      ax.set_xlabel('input_1');
      ax.set_ylabel('input_2');
      ax.set_zlabel('output');
      
    

if __name__=='__main__':
    """Pass a directory argument, read inputs, targets and predictions from that directory, and plot."""

    if (len(sys.argv)==2):
        directory = str(sys.argv[1])
    else:
        print('\nUsage: '+sys.argv[0]+' <directory>    (data is read from directory)\n')
        sys.exit()
    
    fig = plt.figure()                                                            
    if (getDataDimFromDirectory(directory)==1):
      ax = fig.gca()
    else:
      ax = Axes3D(fig)
    plotDataFromDirectory(directory,ax)
    plt.show()


