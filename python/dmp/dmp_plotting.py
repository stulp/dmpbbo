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

from dynamicalsystems.DynamicalSystem import * 

def getDmpAxes(has_fa_output=False):
    n_cols = 5
    n_rows = 3 if has_fa_output else 2
    fig = plt.figure(figsize=(3*n_cols,3*n_rows))
    
    axs = [ fig.add_subplot(n_rows,5,i+1) for i in range(n_rows*5) ]
    return axs

def plotDmp(tau,ts,xs,xds,**kwargs):
    forcing_terms = kwargs.get('forcing_terms',[]) 
    fa_output = kwargs.get('fa_output',[]) 
    ext_dims = kwargs.get('ext_dims',[]) 
    has_fa_output = len(forcing_terms)>0 or len(fa_output)>0
    
    axs = kwargs.get('axs') or getDmpAxes(has_fa_output)
        

    # Dimensionality of dynamical system.
    dim_x = xs.shape[1]      
    # Dimensionality of the DMP. -2 because of phase and gating (which are 1D) and /3 because of spring system (which has dimensionality 2*n_dims_dmp) and goal system (which has dimensionality n_dims_dmp)
    n_dims_dmp = (dim_x-2)//3
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
        cur_n_plots = 2
        if (system_order[i_system]==2):
            cur_n_plots = 3
        
        cur_axs = axs[subplot_offsets[i_system]-1:subplot_offsets[i_system]-1+cur_n_plots]
        cur_indices = list(system_indices[i_system])
        cur_xs = xs[:,cur_indices]
        cur_xds = xds[:,cur_indices]
        if (system_order[i_system]==2):
            lines = DynamicalSystem.plotStatic(tau,ts,cur_xs,cur_xds,axs=cur_axs,dim_y=n_dims_dmp);
        else:
            lines = DynamicalSystem.plotStatic(tau,ts,cur_xs,cur_xds,axs=cur_axs);
            
        if (system_names[i_system]=='gating'):
          plt.setp(lines,color='m')
          cur_axs[0].set_ylim([0, 1.1])
        if (system_names[i_system]=='phase'):
          cur_axs[0].set_ylim([0, 1.1])
          plt.setp(lines,color='c')
          
        for ii in range(len(cur_axs)):
          x = numpy.mean(cur_axs[ii].get_xlim())
          y = numpy.mean(cur_axs[ii].get_ylim())
          cur_axs[ii].text(x,y,system_names[i_system], horizontalalignment='center');
          if (ii==0):
              cur_axs[ii].set_ylabel(r'$'+system_varname[i_system]+'$')
          if (ii==1):
              cur_axs[ii].set_ylabel(r'$\dot{'+system_varname[i_system]+'}$')
          if (ii==2):
              cur_axs[ii].set_ylabel(r'$\ddot{'+system_varname[i_system]+'}$')
        
    # todo Fix this
    if len(fa_output)>1:
        ax = axs[11-1]
        ax.plot(ts,fa_output)
        x = numpy.mean(ax.get_xlim())
        y = numpy.mean(ax.get_ylim())
        ax.text(x,y,'func. approx.', horizontalalignment='center');                                        
        ax.set_xlabel(r'time ($s$)');
        ax.set_ylabel(r'$f_\mathbf{\theta}('+system_varname[0]+')$');
    
    if len(forcing_terms)>1:
        ax = axs[12-1]
        ax.plot(ts,forcing_terms)
        x = numpy.mean(ax.get_xlim())
        y = numpy.mean(ax.get_ylim())
        ax.text(x,y,'forcing term', horizontalalignment='center');                                        
        ax.set_xlabel(r'time ($s$)');
        ax.set_ylabel(r'$v\cdot f_{\mathbf{\theta}}('+system_varname[0]+')$');
    
    if (len(ext_dims)>1):
        ax = axs[13-1]
        ax.plot(ts,ext_dims)
        x = numpy.mean(ax.get_xlim())
        y = numpy.mean(ax.get_ylim())
        ax.text(x,y,'extended dims', horizontalalignment='center');                                        
        ax.set_xlabel(r'time ($s$)');
        ax.set_ylabel(r'unknown');

    x_lim = [min(ts),max(ts)]
    for ax in plt.gcf().get_axes():
        ax.plot([tau,tau],ax.get_ylim(),'-k')
        ax.set_xlim(x_lim[0],x_lim[1])
            


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
      
    x_lim = [min(trajectory[:,time_index]),max(trajectory[:,time_index])]
    for ax in axs:
        ax.set_xlim(x_lim[0],x_lim[1])
        
        
      
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
