from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
import sys

from plotData import getDataDimFromDirectory
from plotData import plotDataFromDirectory
from plotBasisFunctions import plotBasisFunctions

def plotLocallyWeightedLines(inputs,lines,ax,n_samples_per_dim,activations_normalized=[],activations=[]):
    """Plots locally weighted lines, whilst being smart about the dimensionality of input data."""
    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        
        #line_handles = ax.plot(inputs,lines, '-', color='#cccccc')
        #line_handles = ax.plot(inputs,lines_weighted, '-', color='lightblue')

        #y_lim_min = min(lines_weighted) - 0.2*(max(lines_weighted)-min(lines_weighted))
        #y_lim_max = max(lines_weighted) + 0.2*(max(lines_weighted)-min(lines_weighted))
        #ax.set_ylim(y_lim_min,y_lim_max)

        if (len(activations_normalized)>0):
              ax_two = ax.twinx()
              line_handles_bfs =               plotBasisFunctions(inputs,activations,ax_two,n_samples_per_dim)
              plt.setp(line_handles_bfs,color='#aaffaa')
              line_handles_bfs = plotBasisFunctions(inputs,activations_normalized,ax_two,n_samples_per_dim)
              plt.setp(line_handles_bfs,color='green')
              for tl in ax_two.get_yticklabels():
                    tl.set_color('green')
                    
              ax_two.set_ylim(-2.0,3.0)
   
              for ii in xrange(len(activations_normalized[0])):
                    active = activations_normalized[:,ii]>(max(activations_normalized[:,ii])*0.001)
                    line_handles = ax.plot(inputs[active],lines[active,ii], '--',color='#aaaaaa',linewidth='1')
    
    elif (n_dims==2):
        n_lines = len(lines[0])
        # Have a look at the colormaps here and decide which one you'd like:
        # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        colormap = plt.cm.Set3
        colors = [colormap(i) for i in numpy.linspace(0, 0.9, n_lines)]

        inputs_0_on_grid = numpy.reshape(inputs[:,0],n_samples_per_dim)
        inputs_1_on_grid = numpy.reshape(inputs[:,1],n_samples_per_dim)
        if (len(activations)>0):
            for i_line in range(n_lines):
                cur_color = colors[i_line]
                cur_line       = lines[:,i_line];
                cur_activations = activations[:,i_line];
                cur_line[cur_activations<0.4] = numpy.nan # Make plotting easier by leaving out small numbers
                line_on_grid = numpy.reshape(cur_line,n_samples_per_dim)
                line_handles = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,line_on_grid,linewidth=2,rstride=1, cstride=1, color=cur_color)
            
        line_on_grid = numpy.reshape(lines_weighted,n_samples_per_dim)
        line_handles = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,line_on_grid,linewidth=1,rstride=1, cstride=1, color="#333333")
          
    else:
        print 'Cannot plot input data with a dimensionality of '+str(n_dims)+'.'
        line_handles = []
        
    return line_handles


def plotLocallyWeightedLinesFromDirectory(directory,ax,plot_normalized=True):
    """Read activations from file, and plot them."""
  
    try:
      inputs = numpy.loadtxt(directory+'/inputs_grid.txt')                             
    except IOError:
      return False;
      
    lines  = numpy.loadtxt(directory+'/lines.txt')                         
    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims>2):
        sys.exit('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
    
    try:
        n_samples_per_dim = numpy.loadtxt(directory+'/n_samples_per_dim.txt')                             
    except IOError:
        # Assume data is 1D
        n_samples_per_dim = len(inputs)
        
    try:
        activations_normalized = numpy.loadtxt(directory+'/activations_normalized.txt')                             
    except IOError:
        activations_normalized = []
    try:
        activations = numpy.loadtxt(directory+'/activations.txt')                             
    except IOError:
        activations = []
      
    plotLocallyWeightedLines(inputs,lines,ax,n_samples_per_dim,activations_normalized,activations) 

    if (n_dims==1):
      ax.set_xlabel('input');
      ax.set_ylabel('output');
    else:
      ax.set_xlabel('input_1');
      ax.set_ylabel('input_2');
      ax.set_zlabel('output');

    return True;
    
    

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
    plotLocallyWeightedLinesFromDirectory(directory,ax)
    plt.show()


