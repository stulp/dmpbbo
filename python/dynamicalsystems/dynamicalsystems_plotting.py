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


import numpy
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def checkIfVectorContainsTime(ts):
    # Here we check if the first column (extracted above) represents time
    # (i.e. is monotonically increasing, and constant dt
    diff_ts = numpy.diff(ts)
    monotonically_increasing = numpy.all(diff_ts>0)
    constant_dt = numpy.std(diff_ts)<0.000001
    if not monotonically_increasing or not constant_dt:
        print('WARNING: First column does not seem to represent time. Please change the way plotDynamicalSystem is called, or use plotDynamicalSystemDeprecated instead.')


def plotDynamicalSystem(data,axs):
  
    ts  = data[:,0] # First column is time
    data = data[:,1:] # Remove first column
    
    # Here comes a check just for backwards compatibility
    # Input format for data used to be [ x_1..x_D  xd_1..xd_D  t ]
    # Input format for data now is     [ t  x_1..x_D  xd_1..xd_D ]
    checkIfVectorContainsTime(ts)            
   
    # Prepare tex intepretation
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Get dimensionality of the dynamical system
    #     divide by 2 because we have x and xd
    dim = (data.shape[1])//2

    system_order = len(axs)-1

        
    if (system_order==1):
      
        # data has following format: [ x_1..x_D  xd_1..xd_D ]
        xs  = data[:,0*dim:1*dim]
        xds = data[:,1*dim:2*dim]
        
        lines = axs[0].plot(ts,xs)
        axs[0].set_ylabel(r"$x$")
        
        lines[len(lines):] = axs[1].plot(ts,xds)
        axs[1].set_ylabel(r"$\dot{x}$")
        
    else:
        # data has following format: [ y_1..y_D  z_1..z_D   yd_1..yd_D  zd_1..zd_D ]
        
        # For second order systems, dim_orig = dim/2 (because x = [y z] and xd = [yd zd]
        dim_orig =dim//2;
        
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

def plotDynamicalSystemDeprecated(data_time_last,axs):
    # Make last column (time) the first column
    data_time_first = numpy.roll(data_time_last, 1, axis=1)
    return plotDynamicalSystem(data_time_first,axs)
   

def plotDynamicalSystemComparison(data1,data2,name1,name2,axs,axs_diff):

    lines1     = plotDynamicalSystem(data1,axs)
    lines2     = plotDynamicalSystem(data2,axs)
    plt.setp(lines1,linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label=name1)
    plt.setp(lines2,linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label=name2)
    plt.legend()
    
    data_diff = data1-data2;
    data_diff[:,0] = data1[:,0] # Don't subtract time...
    lines_diff = plotDynamicalSystem(data_diff,axs_diff)
    plt.setp(lines_diff,linestyle= '-', linewidth=2, color=(0.50,0.00,0.00), label='diff')
    plt.legend()
        

#if __name__=='__main__':
#  # Process arguments
#  if ( (len(sys.argv)<3)):
#    print('\nUsage: '+sys.argv[0]+' <filename1> <filename2> [system order]\n')
#    sys.exit()
#  
#  filename1 = str(sys.argv[1])
#  filename2 = str(sys.argv[2])
#  
#  system_order = 1
#  if ( (len(sys.argv)>3)):
#    system_order = int(sys.argv[3])
#  
#  try:
#      data1 = numpy.loadtxt(filename1);
#  except IOError:
#      print("File '"+filename1+ "' does not exist. ABORT.")
#      sys.exit(-1)
#  
#  try:
#      data2 = numpy.loadtxt(filename2);
#  except IOError:
#      print("File '"+filename2+ "' does not exist. ABORT.")
#      sys.exit(-1)
#     
#  fig = plt.figure(1,figsize=(12, 4))
#  axs = [];
#  axs_diff = [];
#  for sp in range(2*system_order):
#      axs.append(fig.add_subplot(2,2*system_order,sp+1));
#      axs_diff.append(fig.add_subplot(2,2*system_order,sp+1+2*system_order));
# 
#  plotDynamicalSystemComparison(data1,data2,filename1,filename2,axs,axs_diff)
#  #fig.tight_layout()
#  plt.show()
  
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