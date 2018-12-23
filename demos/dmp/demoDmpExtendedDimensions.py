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


## \file demoDmp.py
## \author Freek Stulp
## \brief  Visualizes results of demoDmp.cpp
## 
## \ingroup Demos
## \ingroup Dmps

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys, subprocess

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)

from dmp.dmp_plotting import * 

if __name__=='__main__':
    
    # Call the executable with the directory to which results should be written
    executable = "../../bin/demoDmpExtendedDimensions"
    directory = "/tmp/demoDmpExtendedDimensions"
    executeBinary(executable, directory)
    
    print("Plotting")
    
    fig = plt.figure(1)
    axs = [ fig.add_subplot(141), fig.add_subplot(142), fig.add_subplot(143), fig.add_subplot(144) ] 
    
    n_misc = 2
    lines = plotTrajectoryFromFile(directory+"/demonstration_traj.txt",axs,n_misc)
    plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')
    
    lines = plotTrajectoryFromFile(directory+"/reproduced_traj.txt",axs,n_misc)
    plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')
    
    plt.legend()
    fig.canvas.set_window_title('Comparison between demonstration and reproduced') 
    
    # Read data
    ts_xs_xds        = numpy.loadtxt(directory+'/reproduced_ts_xs_xds.txt')
    forcing_terms = numpy.loadtxt(directory+'/reproduced_forcing_terms.txt')
    fa_output     = numpy.loadtxt(directory+'/reproduced_fa_output.txt')
    fa_extended   = numpy.loadtxt(directory+'/reproduced_fa_extended.txt')


    fig = plt.figure(2)
    plotDmp(ts_xs_xds,fig,forcing_terms,fa_output,fa_extended)
    fig.canvas.set_window_title('Analytical integration') 
    
    
    
    reproduced_step = numpy.loadtxt(directory+'/reproduced_step_ts_xs_xds_ext.txt')
    ts_xs_xds = reproduced_step[:,:-2]
    fa_ext_step = reproduced_step[:,-2:]
    fig = plt.figure(3)
    plotDmp(ts_xs_xds,fig,[],[],fa_ext_step)
    fig.canvas.set_window_title('Step-by-step integration') 
    
    
    plt.show()
