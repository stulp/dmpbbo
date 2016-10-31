## \file demoDmpContextual.py
## \author Freek Stulp
## \brief  Visualizes results of demoDmpContextual.cpp
## 
## \ingroup Demos
## \ingroup Dmps

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess

lib_path = os.path.abspath('../plotting')
sys.path.append(lib_path)

from plotTrajectory import plotTrajectoryFromFile
from plotDmp import plotDmp

executable = "../../../bin/demoDmpContextual"

if (not os.path.isfile(executable)):
    print("")
    print("ERROR: Executable '"+executable+"' does not exist.")
    print("Please call 'make install' in the build directory first.")
    print("")
    sys.exit(-1);

# Call the executable with the directory to which results should be written
main_directory = "/tmp/demoDmpContextual"

# Test both 1-step and 2-step Dmps
for n_dmp_contextual_step in [1, 2]:
    print("_______________________________________________________________")
    print("Demo for "+str(n_dmp_contextual_step)+"-step contextual Dmp")
    
    directory = main_directory + "/Step"+str(n_dmp_contextual_step)

    command = executable+" "+directory+" "+str(n_dmp_contextual_step)
    print(command)
    subprocess.call(command, shell=True)
    
    print("Plotting")
    
    task_parameters_demos   = numpy.loadtxt(directory+"/task_parameters_demos.txt")
    task_parameters_repros   = numpy.loadtxt(directory+"/task_parameters_repros.txt")
    n_demos = len(task_parameters_demos)
    n_repros = len(task_parameters_repros)
    
    fig = plt.figure(n_dmp_contextual_step)
    fig.suptitle(str(n_dmp_contextual_step)+"-step Contextual Dmp")
    axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 
    
    for i_demo in range(n_demos):
      filename = "demonstration0"+str(i_demo);
      lines = plotTrajectoryFromFile(directory+"/"+filename+".txt",axs)
      plt.setp(lines, linestyle='-',  linewidth=4, color=(0.7,0.7,1.0), label=filename)
    
    for i_repro in range(n_repros):
      filename = "reproduced0"+str(i_repro);
      lines = plotTrajectoryFromFile(directory+"/"+filename+".txt",axs)
      plt.setp(lines, linestyle='-', linewidth=1, color=(0.5,0.0,0.0), label=filename)
    
    if n_dmp_contextual_step==2:
        fig = plt.figure(n_dmp_contextual_step+1)
        n_polpar = 5
        for i_polpar in range(n_polpar):
            ax = fig.add_subplot(1,n_polpar,i_polpar+1)
            dir_polpar = directory+'/dim0_polpar'+str(i_polpar)
            print(dir_polpar)

            inputs   = numpy.loadtxt(dir_polpar+'/inputs.txt')
            targets  = numpy.loadtxt(dir_polpar+'/targets.txt')
            inputs_grid = numpy.loadtxt(dir_polpar+'/inputs_grid.txt')
            pred_grid = numpy.loadtxt(dir_polpar+'/predictions_grid.txt')
    
            ax.plot(inputs,targets,'o',color='green',label='targets')
            ax.plot(inputs_grid,pred_grid,linestyle='-',color='red',label='predictions')
            ax.set_xlabel('task parameter')
            ax.set_ylabel('policy parameter')
            plt.legend()
    
    #ax = fig.add_subplot(155)
    #for i_repro in range(n_repros):
    #  filename = directory+"/"+"reproduced_forcingterm0"+str(i_repro)+".txt";
    #  forcingterm = numpy.loadtxt(filename);
    #  lines = ax.plot(numpy.linspace(0,1,len(forcingterm)),forcingterm)
    #  plt.setp(lines, linestyle='-', linewidth=2, color=(0.0,0.0,0.5), label=filename)
    #ax.set_xlabel('phase')
    #ax.set_ylabel('forcing term')
      
    
plt.show()
