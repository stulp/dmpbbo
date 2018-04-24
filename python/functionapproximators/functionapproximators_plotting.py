# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


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

def plotBasisFunctions(inputs,activations,ax,n_samples_per_dim,cosine_basis_functions=False):
    """Plot basis functions activations, whilst being smart about the dimensionality of input data."""
    
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        lines = ax.plot(inputs,activations, '-')
        
    elif (n_dims==2):
        n_basis_functions = len(activations[0])
        basis_functions_to_plot = range(n_basis_functions)
        
        #min_val = numpy.amin(activations)
        if (cosine_basis_functions):
            # Dealing with cosine basis functions here, only plot the 3 with the heightest value
            max_vals = numpy.amax(activations,axis=0)
            indices = numpy.argsort(max_vals)
            n_to_plot = min([n_basis_functions, 3])
            basis_functions_to_plot = indices[-n_to_plot:]

        # Have a look at the colormaps here and decide which one you'd like:
        # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        colormap = plt.cm.Set1
        colors = [colormap(i) for i in numpy.linspace(0, 0.9, n_basis_functions)]

        inputs_0_on_grid = numpy.reshape(inputs[:,0],n_samples_per_dim)
        inputs_1_on_grid = numpy.reshape(inputs[:,1],n_samples_per_dim)
        lines = [];
        values_range = numpy.amax(activations)-numpy.amin(activations)
        for i_basis_function in basis_functions_to_plot:
          cur_color = colors[i_basis_function]
          cur_activations = activations[:,i_basis_function];
          # Make plotting easier by leaving out small numbers
          cur_activations[numpy.abs(cur_activations)<values_range*0.001] = numpy.nan 
          activations_on_grid = numpy.reshape(cur_activations,n_samples_per_dim)
          cur_lines = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,activations_on_grid,linewidth=0.5,rstride=1, cstride=1, color=cur_color)
          lines.append(cur_lines) 
          
    else:
        print('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
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
    
    cosine_basis_functions = False

    try:
        filename = directory+'/n_samples_per_dim.txt'
        n_samples_per_dim = numpy.loadtxt(filename,dtype=int)                             
    except IOError:
        # Assume data is 1D
        n_samples_per_dim = len(inputs)
        
    #try:
    #    predictions  = numpy.loadtxt(directory+'/predictions_grid.txt')
    #    plotDataPredictionsGrid(inputs,predictions,ax,n_samples_per_dim)
    #except IOError:
    #    predictions = [];
        
    lines_bfs = None
    if (n_dims<2):
        try:
            activations  = numpy.loadtxt(directory+'/activations_grid.txt')
            lines_bfs = plotBasisFunctions(inputs,activations,ax,n_samples_per_dim,cosine_basis_functions) 
        except IOError:
            activations = None

    lines_bfs_weighted = None;
    try:
        activations_weighted  = numpy.loadtxt(directory+'/activations_weighted_grid.txt')
        lines_bfs_weighted = plotBasisFunctions(inputs,activations_weighted,ax,n_samples_per_dim,cosine_basis_functions) 
    except IOError:
        activations_weighted_grid = None;

    if (n_dims==1):
        if lines_bfs_weighted is not None and lines_bfs is not None:
            # Both are plotted, give different colors
            plt.setp(lines_bfs_weighted,color='green')
            plt.setp(lines_bfs,color='#aaffaa')
        else:
            if lines_bfs_weighted is not None:
                plt.setp(lines_bfs,color='green')
            if lines_bfs != None:
                plt.setp(lines_bfs,color='green')

    if (n_dims==1):
      ax.set_xlabel('input');
      ax.set_ylabel('activation');
    else:
      ax.set_xlabel('input_1');
      ax.set_ylabel('input_2');
      ax.set_zlabel('activation');
      
    return True

#if __name__=='__main__':
#    """Pass a directory argument, read inputs, targets and predictions from that directory, and plot."""
#
#    if (len(sys.argv)==2):
#        directory = str(sys.argv[1])
#    else:
#        print('\nUsage: '+sys.argv[0]+' <directory>    (data is read from directory)\n')
#        sys.exit()
#    
#    fig = plt.figure(1) 
#    if (getDataDimFromDirectory(directory)==1):
#      ax = fig.gca()
#    else:
#      ax = Axes3D(fig)
#
#    plotBasisFunctionsFromDirectory(directory,ax)
#    fig.gca().set_title('Normalized activations')
#    
#    #fig = plt.figure(2) 
#    #if (getDataDimFromDirectory(directory)==1):
#    #  ax = fig.gca()
#    #else:
#    #  ax = Axes3D(fig)
#    #  
#    #plot_normalized = True;
#    #plotBasisFunctionsFromDirectory(directory,ax,plot_normalized)
#    #fig.gca().set_title('Activations')
#    
#    
#      
#    plt.show()


def plotLocallyWeightedLines(inputs,lines,ax,n_samples_per_dim,activations=None,activations_unnormalized=None):
    """Plots locally weighted lines, whilst being smart about the dimensionality of input data."""
    
    line_handles = []
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims==1):
        
        if not activations is None:
            ax_two = ax.twinx()
            
            # Plot basis functions
            if not activations_unnormalized is None:
                line_handles_bfs = plotBasisFunctions(inputs,activations_unnormalized,ax_two,n_samples_per_dim)
                plt.setp(line_handles_bfs,color='#aaffaa')
            line_handles_bfs = plotBasisFunctions(inputs,activations,ax_two,n_samples_per_dim)
            plt.setp(line_handles_bfs,color='green')
            for tl in ax_two.get_yticklabels():
                tl.set_color('green')
            
            ax_two.set_ylim(-2.0,3.0)
            
            # Plot line segements
            n_basis_functions =  len(numpy.atleast_1d(activations[0]));
            if n_basis_functions==1:
              active = activations>(max(activations)*0.001)
              line_handles = ax.plot(inputs[active],lines[active], '--',color='#aaaaaa',linewidth=0.5)
            else:
              for ii in range(n_basis_functions):
                  active = activations[:,ii]>(max(activations[:,ii])*0.0001)
                  line_handles = ax.plot(inputs[active],lines[active,ii], '--',color='#aaaaaa',linewidth=1)
    
    elif (n_dims==2):
        inputs_0_on_grid = numpy.reshape(inputs[:,0],n_samples_per_dim)
        inputs_1_on_grid = numpy.reshape(inputs[:,1],n_samples_per_dim)
        
        
        n_lines = len(lines[0])
        # Have a look at the colormaps here and decide which one you'd like:
        # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        colormap = plt.cm.Set1
        shuffler = numpy.linspace(0, 0.9, n_lines)
        random.shuffle(shuffler)
        colors = [colormap(i) for i in shuffler]


        if (len(activations)>0):
            min_val = numpy.amax(lines)
            for i_line in range(n_lines):
                cur_color = colors[i_line]
                cur_line       = lines[:,i_line];
                cur_activations = activations[:,i_line];
                cur_line[cur_activations<0.25] = numpy.nan # Make plotting easier by leaving out small numbers
                line_on_grid = numpy.reshape(cur_line,n_samples_per_dim)
                line_handles = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,line_on_grid,linewidth=0.5,rstride=1, cstride=1, color=cur_color)
                
                if numpy.nanmin(cur_line)<min_val:
                    min_val = numpy.nanmin(cur_line)
            
            level = numpy.mean([numpy.amin(activations), numpy.amax(activations)])
            for i_line in range(n_lines):
                cur_color = colors[i_line]
                cur_activations = activations[:,i_line];
                acts_on_grid = numpy.reshape(cur_activations,n_samples_per_dim)
                cset = ax.contour(inputs_0_on_grid, inputs_1_on_grid, acts_on_grid, [level], zdir='z', offset=min_val, colors=[cur_color])
                
                
        #line_on_grid = numpy.reshape(lines_weighted,n_samples_per_dim)
        #line_handles = ax.plot_wireframe(inputs_0_on_grid,inputs_1_on_grid,line_on_grid,linewidth=1,rstride=1, cstride=1, color="#333333")
          
    else:
        print('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
        line_handles = []
        
    return line_handles


def plotLocallyWeightedLinesFromDirectory(directory,ax):
    """Read activations from file, and plot them."""
  
    try:
      inputs = numpy.loadtxt(directory+'/inputs_grid.txt')                             
    except IOError:
      return False;
      
    try:
        lines  = numpy.loadtxt(directory+'/lines_grid.txt')                         
    except IOError:
        # If there are no lines, assume this to be a weighted sum of basis functions instead
        return plotBasisFunctionsFromDirectory(directory,ax)
      
    n_dims = len(numpy.atleast_1d(inputs[0]))
    if (n_dims>2):
        sys.exit('Cannot plot input data with a dimensionality of '+str(n_dims)+'.')
    
    try:
        filename = directory+'/n_samples_per_dim.txt'
        n_samples_per_dim = numpy.loadtxt(filename,dtype=int)
    except IOError:
        # Assume data is 1D
        n_samples_per_dim = len(inputs)
        
    #try:
    #    predictions  = numpy.loadtxt(directory+'/predictions_grid.txt')
    #    plotDataPredictionsGrid(inputs,predictions,ax,n_samples_per_dim)
    #except IOError:
    #    predictions = [];

    try:
        activations_unnormalized = numpy.loadtxt(directory+'/activations_unnormalized_grid.txt')
    except IOError:
        activations_unnormalized = None
        
    try:
        activations = numpy.loadtxt(directory+'/activations_grid.txt')
    except IOError:
        activations = None
      
        
    plotLocallyWeightedLines(inputs,lines,ax,n_samples_per_dim,activations,activations_unnormalized) 

    if (n_dims==1):
      ax.set_xlabel('input');
      ax.set_ylabel('output');
    else:
      ax.set_xlabel('input_1');
      ax.set_ylabel('input_2');
      ax.set_zlabel('output');

    return True;
    
    

#if __name__=='__main__':
#    """Pass a directory argument, read inputs, targets and predictions from that directory, and plot."""
#
#    if (len(sys.argv)==2):
#        directory = str(sys.argv[1])
#    else:
#        print('\nUsage: '+sys.argv[0]+' <directory>    (data is read from directory)\n')
#        sys.exit()
#    
#  
#    fig = plt.figure() 
#    if (getDataDimFromDirectory(directory)==1):
#      ax = fig.gca()
#    else:
#      ax = Axes3D(fig)
#      
#    plotDataFromDirectory(directory,ax)
#    plotLocallyWeightedLinesFromDirectory(directory,ax)
#    plt.show()


