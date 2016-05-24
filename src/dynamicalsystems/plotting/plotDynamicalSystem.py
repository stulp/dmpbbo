import numpy
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import subprocess

def plotDynamicalSystem(data,axs):
  
    # Prepare tex intepretation
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Get dimensionality of the dynamical system
    #     (-1 to subtract time)/divide by 2 because we have x and xd
    dim = (data.shape[1]-1)/2;

    system_order = len(axs)-1

    if (system_order==1):
      
        # data has following format: [ x_1..x_D  xd_1..xd_D  t ]
        ts  = data[:,-1] # Last column is time
        xs  = data[:,0*dim:1*dim]
        xds = data[:,1*dim:2*dim]
        
        lines = axs[0].plot(ts,xs)
        axs[0].set_ylabel(r"$x$")
        
        lines[len(lines):] = axs[1].plot(ts,xds)
        axs[1].set_ylabel(r"$\dot{x}$")
        
    else:
        # data has following format: [ y_1..y_D  z_1..z_D   yd_1..yd_D  zd_1..zd_D  t ]
        
        # For second order systems, dim_orig = dim/2 (because x = [y z] and xd = [yd zd]
        dim_orig =dim/2;
        
        # data has following format: [ x_1..x_D  xd_1..xd_D  t ]
        ts  = data[:,-1] # Last column is time
        ys  = data[:,0*dim_orig:1*dim_orig]
        zs  = data[:,1*dim_orig:2*dim_orig]
        yds = data[:,2*dim_orig:3*dim_orig]
        zds = data[:,3*dim_orig:4*dim_orig]
        
        lines = axs[0].plot(ts,ys)
        axs[0].set_ylabel(r"$y$")
        
        #lines[len(lines):] = axs[0].plot(ts,zs)
        #axs[0].set_ylabel("z")
        
        lines[len(lines):] = axs[1].plot(ts,yds)
        axs[1].set_ylabel(r"$\dot{y} = z/\tau$")
        
        # Avoid division by zero by ignoring first/last values yds is zero
        xs_yds = zs[3:-3,:]/yds[3:-3,:]
        tau = xs_yds.mean()
        
        lines[len(lines):] = axs[2].plot(ts,zds/tau)
        axs[2].set_ylabel(r"$\ddot{y} = \dot{z}/\tau$")

    for ax in axs:
        ax.set_xlabel(r'time ($s$)')
        #ax.axis('tight')
        ax.grid()
        
    return lines

if __name__=='__main__':
  # Process arguments
  if ( (len(sys.argv)<2)):
    print('\nUsage: '+sys.argv[0]+' <filename> [system order]\n')
    sys.exit()
  
  filename = str(sys.argv[1])
  
  system_order = 1
  if ( (len(sys.argv)>2)):
    system_order = int(sys.argv[2])
  
  try:
      data = numpy.loadtxt(filename);
  except IOError:
      print("File '"+filename+ "' does not exist. ABORT.")
      sys.exit(-1)
     
  fig = plt.figure(1,figsize=(12, 4))
  fig.suptitle(filename)
  axs = [];
  for sp in range(2*system_order):
      axs.append(fig.add_subplot(1,2*system_order,sp+1));
  lines = plotDynamicalSystem(data,axs)
  plt.setp(lines, linewidth=2)
  #fig.tight_layout()
  plt.show()