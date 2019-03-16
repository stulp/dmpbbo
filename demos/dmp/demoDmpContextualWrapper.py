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


## \file demoDmpContextual.py
## \author Freek Stulp
## \brief  Visualizes results of demoDmpContextual.cpp
## 
## \ingroup Demos
## \ingroup Dmps

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)
from dmp.dmp_plotting import * 

if __name__=='__main__':

    
    # Test both 1-step and 2-step Dmps
    for n_dmp_contextual_step in [1, 2]:
        print("_______________________________________________________________")
        print("Demo for "+str(n_dmp_contextual_step)+"-step contextual Dmp")
        
        # Call the executable with the directory to which results should be written
        executable = "./demoDmpContextual"
        main_directory = "/tmp/demoDmpContextual"
        directory = main_directory + "/Step"+str(n_dmp_contextual_step)
        arguments = directory+" "+str(n_dmp_contextual_step)
        executeBinary(executable,arguments,True)
        
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
