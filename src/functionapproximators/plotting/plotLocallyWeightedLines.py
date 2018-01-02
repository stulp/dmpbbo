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


from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy                                                                  
import matplotlib.pyplot as plt                                               
from matplotlib import cm
import sys
import random

from plotData import *
from plotBasisFunctions import plotBasisFunctions
from plotBasisFunctions import plotBasisFunctionsFromDirectory

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
    plotLocallyWeightedLinesFromDirectory(directory,ax)
    plt.show()


