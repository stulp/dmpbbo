import numpy
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import subprocess

from plotDynamicalSystem import plotDynamicalSystem

def plotDynamicalSystemComparison(data1,data2,name1,name2,axs,axs_diff):

    lines1     = plotDynamicalSystem(data1,axs)
    lines2     = plotDynamicalSystem(data2,axs)
    plt.setp(lines1,linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label=name1)
    plt.setp(lines2,linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label=name2)
    plt.legend()
    
    data_diff = data1-data2;
    data_diff[:,-1] = data1[:,-1] # Don't subtract time...
    lines_diff = plotDynamicalSystem(data_diff,axs_diff)
    plt.setp(lines_diff,linestyle= '-', linewidth=2, color=(0.50,0.00,0.00), label='diff')
    plt.legend()
        

if __name__=='__main__':
  # Process arguments
  if ( (len(sys.argv)<3)):
    print('\nUsage: '+sys.argv[0]+' <filename1> <filename2> [system order]\n')
    sys.exit()
  
  filename1 = str(sys.argv[1])
  filename2 = str(sys.argv[2])
  
  system_order = 1
  if ( (len(sys.argv)>3)):
    system_order = int(sys.argv[3])
  
  try:
      data1 = numpy.loadtxt(filename1);
  except IOError:
      print("File '"+filename1+ "' does not exist. ABORT.")
      sys.exit(-1)
  
  try:
      data2 = numpy.loadtxt(filename2);
  except IOError:
      print("File '"+filename2+ "' does not exist. ABORT.")
      sys.exit(-1)
     
  fig = plt.figure(1,figsize=(12, 4))
  axs = [];
  axs_diff = [];
  for sp in range(2*system_order):
      axs.append(fig.add_subplot(2,2*system_order,sp+1));
      axs_diff.append(fig.add_subplot(2,2*system_order,sp+1+2*system_order));

  plotDynamicalSystemComparison(data1,data2,filename1,filename2,axs,axs_diff)
  #fig.tight_layout()
  plt.show()