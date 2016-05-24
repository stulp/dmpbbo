import sys, os
import numpy                                                                    
import matplotlib.pyplot as plt

# Include scripts for plotting
lib_path = os.path.abspath('../../dynamicalsystems/plotting')
sys.path.append(lib_path)
from plotDynamicalSystem import plotDynamicalSystem
from plotDynamicalSystemComparison import plotDynamicalSystemComparison

from plotDynamicalSystem import plotDynamicalSystem

def plotDmp(data,fig,forcing_terms_data=0,fa_output_data=0):

    # Number of columns in the data
    n_cols = data.shape[1]             
    # Dimensionality of dynamical system. -1 because of time, /2 because both x and xd are stored.
    n_state_length = (n_cols-1)//2      
    # Dimensionality of the DMP. -2 because of phase and gating (which are 1D) and /3 because of spring system (which has dimensionality 2*n_dims_dmp) and goal system (which has dimensionality n_dims_dmp)
    n_dims_dmp = (n_state_length-2)//3
    D = n_dims_dmp  # Abbreviation for convencience
    
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
          
        
        indices = list(system_indices[i_system])
        indices_xd =[i+n_state_length for i in indices] # +n_state_length because xd is in second half
        indices.extend(indices_xd) # For derivative
        indices.append(-1) # For time
      
        lines = plotDynamicalSystem(data[:,indices],axs);
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
        
    ts = data[:,-1];

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

