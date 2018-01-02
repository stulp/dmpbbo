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
import numpy as np
import matplotlib.pyplot as plt

from bbo_plotting import plotLearningCurve, plotExplorationCurve
from bbo_plotting import loadLearningCurve, loadExplorationCurve

if __name__=='__main__':
    
    # See if input directory was passed
    if (len(sys.argv)<2):
        print('\nUsage: '+sys.argv[0]+' <directory>\n')
        sys.exit()
        
    directory = str(sys.argv[1])
    exploration_curve = loadExplorationCurve(directory)
    learning_curve = loadLearningCurve(directory)
    
    if exploration_curve is not None: # Plot exploration too?
        fig = plt.figure(1,figsize=(16, 6))
        plotExplorationCurve(exploration_curve,fig.add_subplot(121))
        plotLearningCurve(learning_curve,fig.add_subplot(122))
    else:
        fig = plt.figure(1,figsize=(8, 6))
        plotLearningCurve(learning_curve,fig.add_subplot(111))
        
    plt.show()
