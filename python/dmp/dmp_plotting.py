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


import sys, os
import numpy                                                                    
import matplotlib.pyplot as plt

# Include scripts for plotting
lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)
from dynamicalsystems.dynamicalsystems_plotting import * 


def plotDmp(data,fig,forcing_terms_data=[],fa_output_data=[],ext_dims=[]):

    ts  = data[:,0] # First column is time
    data = data[:,1:] # Remove first column
    
    # Here comes a check just for backwards compatibility
    # Input format for data used to be [ x_1..x_D  xd_1..xd_D  t ]
    # Input format for data now is     [ t  x_1..x_D  xd_1..xd_D ]
    checkIfVectorContainsTime(ts)            
    
    
    # Number of columns in the data
    n_cols = data.shape[1]             
    # Dimensionality of dynamical system. /2 because both x and xd are stored.
    n_state_length = n_cols//2      
    # Dimensionality of the DMP. -2 because of phase and gating (which are 1D) and /3 because of spring system (which has dimensionality 2*n_dims_dmp) and goal system (which has dimensionality n_dims_dmp)
    n_dims_dmp = (n_state_length-2)//3
    D = n_dims_dmp  # Abbreviation for convencience

    #define SPRING    segment(0*dim_orig()+0,2*dim_orig())
    #define SPRING_Y  segment(0*dim_orig()+0,dim_orig())
    #define SPRING_Z  segment(1*dim_orig()+0,dim_orig())
    #define GOAL      segment(2*dim_orig()+0,dim_orig())
    #define PHASE     segment(3*dim_orig()+0,       1)
    #define GATING    segment(3*dim_orig()+1,       1)

    # We will loop over each of the subsystems of the DMP: prepare some variables here
    # Names of each of the subsystems
    system_names   = ['phase','gating','goal','spring'];
    system_varname = [    'x',     'v',  '\mathbf{y}^{g_d}',  '\mathbf{y}' ];
    # The indices they have in the data 
    system_indices = [ range(3*D,3*D+1), range(3*D+1,3*D+2), range(2*D,3*D), range(0*D,2*D) ];
    system_order   = [       1,          1,           1,         2 ];
    # The subplot in which they are plotted (x is plotted here, xd in the subplot+1)
    subplot_offsets = [      1,          6,           3,         8  ];
    
    # Loop over each of the subsystems of the DMP
    n_systems = len(system_names)
    for i_system in range(n_systems):
      
        # Plot 'x' for this subsystem (analytical solution and step-by-step integration)
        #fig.suptitle(filename)
        axs = [];
        axs.append(fig.add_subplot(3,5,subplot_offsets[i_system]))
        axs.append(fig.add_subplot(3,5,subplot_offsets[i_system]+1))
        if (system_order[i_system]==2):
          axs.append(fig.add_subplot(3,5,subplot_offsets[i_system]+2))
          
        
        indices_xs = list(system_indices[i_system])
        indices_xds =[i+n_state_length for i in indices_xs] # +n_state_length because xd is in second half
      
        plot_data =  numpy.concatenate((numpy.atleast_2d(ts).T,data[:,indices_xs],data[:,indices_xds]),axis=1)
        lines = plotDynamicalSystem(plot_data,axs);
        if (system_names[i_system]=='gating'):
          plt.setp(lines,color='m')
          axs[0].set_ylim([0, 1.1])
        if (system_names[i_system]=='phase'):
          axs[0].set_ylim([0, 1.1])
          plt.setp(lines,color='c')
          
        for ii in range(len(axs)):
          x = numpy.mean(axs[ii].get_xlim())
          y = numpy.mean(axs[ii].get_ylim())
          axs[ii].text(x,y,system_names[i_system], horizontalalignment='center');
          if (ii==0):
              axs[ii].set_ylabel(r'$'+system_varname[i_system]+'$')
          if (ii==1):
              axs[ii].set_ylabel(r'$\dot{'+system_varname[i_system]+'}$')
          if (ii==2):
              axs[ii].set_ylabel(r'$\ddot{'+system_varname[i_system]+'}$')
        
    # todo Fix this
    if (len(fa_output_data)>1):
        ax = fig.add_subplot(3,5,11)
        ax.plot(ts,fa_output_data)
        x = numpy.mean(ax.get_xlim())
        y = numpy.mean(ax.get_ylim())
        ax.text(x,y,'func. approx.', horizontalalignment='center');                                        
        ax.set_xlabel(r'time ($s$)');
        ax.set_ylabel(r'$f_\mathbf{\theta}('+system_varname[0]+')$');
    
    if (len(forcing_terms_data)>1):
        ax = fig.add_subplot(3,5,12)
        ax.plot(ts,forcing_terms_data)
        x = numpy.mean(ax.get_xlim())
        y = numpy.mean(ax.get_ylim())
        ax.text(x,y,'forcing term', horizontalalignment='center');                                        
        ax.set_xlabel(r'time ($s$)');
        ax.set_ylabel(r'$v\cdot f_{\mathbf{\theta}}('+system_varname[0]+')$');
    
    if (len(ext_dims)>1):
        ax = fig.add_subplot(3,5,13)
        ax.plot(ts,ext_dims)
        x = numpy.mean(ax.get_xlim())
        y = numpy.mean(ax.get_ylim())
        ax.text(x,y,'extended dims', horizontalalignment='center');                                        
        ax.set_xlabel(r'time ($s$)');
        ax.set_ylabel(r'unknown');


if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)==3):
      directory = str(sys.argv[1])
      filename = str(sys.argv[2])
    else:
      print('\nUsage: '+sys.argv[0]+' <directory> <filename>\n')
      sys.exit()
    
    # Read data
    dynsys_data = numpy.loadtxt(directory+'/'+filename)
    # todo Make robust towards non-existence
    forcing_terms_data = numpy.loadtxt(directory+'/forcing_terms_'+filename)
    fa_output_data = numpy.loadtxt(directory+'/fa_output_'+filename)
    
    fig = plt.figure(1,figsize=(15, 8))
    plotDmp(dynsys_data,fig,forcing_terms_data,fa_output_data)
    
    plt.tight_layout()
    plt.show()                                                                      


def plotTrajectory(trajectory,axs,n_misc=0):
    """Plot a trajectory"""
    n_dims = (len(trajectory[0])-1-n_misc)//3 
    # -1 for time, /3 because contains y,yd,ydd
    time_index = 0;
    lines = axs[0].plot(trajectory[:,time_index],trajectory[:,1:n_dims+1], '-')
    axs[0].set_xlabel('time (s)');
    axs[0].set_ylabel('y');
    if (len(axs)>1):
      lines[len(lines):] = axs[1].plot(trajectory[:,time_index],trajectory[:,n_dims+1:2*n_dims+1], '-')
      axs[1].set_xlabel('time (s)');
      axs[1].set_ylabel('yd');
    if (len(axs)>2):
      lines[len(lines):] = axs[2].plot(trajectory[:,time_index],trajectory[:,2*n_dims+1:3*n_dims+1], '-')
      axs[2].set_xlabel('time (s)');
      axs[2].set_ylabel('ydd');
      
    if n_misc>0 and len(axs)>3:
      lines[len(lines):] = axs[3].plot(trajectory[:,time_index],trajectory[:,3*n_dims+1:], '-')
      axs[3].set_xlabel('time (s)');
      axs[3].set_ylabel('misc');
        
      
    return lines

def plotTrajectoryFromFile(filename,axs,n_misc=0):
    """Read trajectory, and plot it."""
    # Read data
    trajectory   = numpy.loadtxt(filename)
    lines = plotTrajectory(trajectory,axs,n_misc)
    return lines

#if __name__=='__main__':
#    """Pass a filename argument, read trajectory, and plot."""
#
#    if (len(sys.argv)==2):
#        filename = str(sys.argv[1])
#    else:
#        print('\nUsage: '+sys.argv[0]+' <filename>    (trajectory is read from file)\n')
#        sys.exit()
#        
#    fig = plt.figure()                                                            
#    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
#    lines = plotTrajectoryFromFile(filename,axs)
#    plt.setp(lines, linewidth=2)
#    plt.show()
