from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import sys

from plotData import getDataDimFromDirectory
from plotData import plotDataPredictionsGrid

def plotBasisFunctions(inputs,activations,ax,n_samples_per_dim):
    """Plot basis functions activations, whilst being smart about the dimensionality of input data."""
    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        lines = ax.plot(inputs,activations, '-')
        
    elif (n_dims==2):
        n_basis_functions = len(activations[0])
        # Have a look at the colormaps here and decide which one you'd like:
        # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        colormap = plt.cm.Set3
        colors = [colormap(i) for i in numpy.linspace(0, 0.9, n_basis_functions)]

        inputs_0_on_grid = numpy.reshape(inputs[:,0],n_samples_per_dim)
        inputs_1_on_grid = numpy.reshape(inputs[:,1],n_samples_per_dim)
        for i_basis_function in range(n_basis_functions):
          cur_color = colors[i_basis_function]
          cur_activations = activations[:,i_basis_function];
          cur_activations[cur_activations<0.001] = numpy.nan # Make plotting easier by leaving out small numbers
          activations_on_grid = numpy.reshape(cur_activations,n_samples_per_dim)
          lines = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,activations_on_grid,linewidth=1,rstride=1, cstride=1, color=cur_color)
          
    else:
        print 'Cannot plot input data with a dimensionality of '+str(n_dims)+'.'
        lines = []
        
    return lines


def plotBasisFunctionsFromDirectory(directory,ax):
    """Read activations from file, and plot them."""
  
    # Relevant files
    #   n_samples_per_dim.txt
    #   inputs_grid.txt
    #   activations_grid.txt
    #   activations_weighted_grid.txt
    #   predictions_grid.txt

    inputs   = numpy.loadtxt(directory+'/inputs_grid.txt')    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims>2):
        sys.exit('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
    

    try:
        n_samples_per_dim = numpy.loadtxt(directory+'/n_samples_per_dim.txt')                             
    except IOError:
        # Assume data is 1D
        n_samples_per_dim = len(inputs)
        
    try:
        predictions  = numpy.loadtxt(directory+'/predictions_grid.txt')
        plotDataPredictionsGrid(inputs,predictions,ax,n_samples_per_dim)
    except IOError:
        predictions = [];
        
    try:
        activations  = numpy.loadtxt(directory+'/activations_grid.txt')
        lines_bfs = plotBasisFunctions(inputs,activations,ax,n_samples_per_dim) 
    except IOError:
        activations = [];
        lines_bfs = [];

    try:
        activations_weighted  = numpy.loadtxt(directory+'/activations_weighted_grid.txt')
        lines_bfs_weighted = plotBasisFunctions(inputs,activations_weighted,ax,n_samples_per_dim) 
        plt.setp(lines_bfs,color='#aaffaa')
        plt.setp(lines_bfs_weighted,color='green')
    except IOError:
        activations_weighted_grid = [];
        lines_bfs_weighted = [];
        plt.setp(lines_bfs,color='green')
        

    if (n_dims==1):
      ax.set_xlabel('input');
      ax.set_ylabel('activation');
    else:
      ax.set_xlabel('input_1');
      ax.set_ylabel('input_2');
      ax.set_zlabel('activation');
      
    

if __name__=='__main__':
    """Pass a directory argument, read inputs, targets and predictions from that directory, and plot."""

    if (len(sys.argv)==2):
        directory = str(sys.argv[1])
    else:
        print '\nUsage: '+sys.argv[0]+' <directory>    (data is read from directory)\n';
        sys.exit()
    
    fig = plt.figure(1) 
    if (getDataDimFromDirectory(directory)==1):
      ax = fig.gca()
    else:
      ax = Axes3D(fig)

    plotBasisFunctionsFromDirectory(directory,ax)
    fig.gca().set_title('Normalized activations')
    
    #fig = plt.figure(2) 
    #if (getDataDimFromDirectory(directory)==1):
    #  ax = fig.gca()
    #else:
    #  ax = Axes3D(fig)
    #  
    #plot_normalized = True;
    #plotBasisFunctionsFromDirectory(directory,ax,plot_normalized)
    #fig.gca().set_title('Activations')
    
    
      
    plt.show()


