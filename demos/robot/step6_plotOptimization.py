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


import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

lib_path = os.path.abspath('../python')
sys.path.append(lib_path)

from dmp_bbo.Task import Task
from dmp_bbo.dmp_bbo_plotting import *
from dmp_bbo.run_one_update_json import *

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="directory to read results from")
    args = parser.parse_args()

    session = LearningSession(args.directory)
    n_subplots = 4
    fig = plt.figure(3,figsize=(n_subplots*3,3))
    axs = [fig.add_subplot(1,n_subplots,1+i) for i in range(n_subplots)]
    session.plotDistributionUpdates(axs[0])
    session.plotRollouts(axs[1])
    session.plotExplorationCurve(axs[2])
    session.plotLearningCurve(axs[3])
    
    plt.show()

