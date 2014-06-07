from mpl_toolkits.mplot3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import sys

# 
def plotData(inputs,outputs,ax):
    """Plot outputs against inputs, whilst being smart about the dimensionality of inputs."""
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        lines = ax.plot(inputs,outputs, '.')
    elif (n_dims==2):
        lines = ax.plot(inputs[:,0],inputs[:,1],outputs, '.')
    else:
        print 'Cannot plot input data with a dimensionality of '+str(n_dims)+'.'
        lines = []
        
    return lines


def plotDataTargets(inputs,target,ax):
    """Plot outputs against targets, and apply default style for targets."""
    lines = plotData(inputs,target,ax)
    plt.setp(lines, linestyle='.', label='targets', color='black',markersize=7)                  
    return lines


def plotDataPredictions(inputs,predictions,ax,n_samples_per_dim=[]):
    """Plot outputs against targets, and apply default style for predictions."""
    lines = plotData(inputs,predictions,ax)
    plt.setp(lines, marker='.', label='predictions', color='red')
    return lines
    
def plotResiduals(inputs,targets,predictions,ax):
    lines = []
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims>2):
        print 'Cannot plot input data with a dimensionality of '+str(n_dims)+'.'
        return lines
    for ii in range(len(inputs)):
        if (n_dims==1):
            lines = ax.plot([inputs[ii], inputs[ii]],[targets[ii], predictions[ii] ], '-r',linewidth=2,label='residuals')
        elif (n_dims==2):
            lines = ax.plot([inputs[ii,0], inputs[ii,0]],[inputs[ii,1], inputs[ii,1]],[targets[ii], predictions[ii] ], '-r',linewidth=2,label='residuals')

    
def plotDataPredictionsGrid(inputs,predictions,ax,n_samples_per_dim=[]):
    """Plot outputs against targets, and apply default style for predictions."""
    lines = plotData(inputs,predictions,ax)
    plt.setp(lines, linestyle='-', marker=None, label='latent function', color='lightblue',linewidth=3)
    return lines
    
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
    predictions = numpy.loadtxt(directory+'/outputs.txt')
        
    # Plotting
    try:
      inputs_grid = numpy.loadtxt(directory+'/inputs_grid.txt')
      predictions_grid = numpy.loadtxt(directory+'/outputs_grid.txt')
      plotDataPredictionsGrid(inputs_grid,predictions_grid,ax)   
    except IOError:
      # Everything's fine: user did not store grid predictions
      predictions_grid = [];
      
      
    plotResiduals(inputs,targets,predictions,ax)
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
        print '\nUsage: '+sys.argv[0]+' <directory>    (data is read from directory)\n';
        sys.exit()
    
    fig = plt.figure()                                                            
    if (getDataDimFromDirectory(directory)==1):
      ax = fig.gca()
    else:
      ax = Axes3D(fig)
    plotDataFromDirectory(directory,ax)
    plt.show()


