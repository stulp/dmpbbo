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


import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from dmp_bbo_plotting import plotOptimizationRollouts

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)==2):
      directory = str(sys.argv[1])
    else:
      print('\nUsage: '+sys.argv[0]+' <directory>\n')
      sys.exit()
    
    fig = plt.figure()
    plot_rollout = None
    plotOptimizationRollouts(directory,fig,plot_rollout)
    plt.show()

