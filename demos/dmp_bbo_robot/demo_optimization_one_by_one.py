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
import numpy as np

lib_path = os.path.abspath('../../python/')
sys.path.append(lib_path)


from demo_perform_rollouts import performRolloutsFakeRobot
from demo_one_update import oneUpdate



if __name__=="__main__":
    
    directory="/tmp/demo_optimization_one_by_one_python"

    oneUpdate(directory)
    performRolloutsFakeRobot(directory+'/update00000')

    oneUpdate(directory)
    performRolloutsFakeRobot(directory+'/update00001')

    plot_results = True
    oneUpdate(directory,plot_results)
    performRolloutsFakeRobot(directory+'/update00002')

    oneUpdate(directory)
    performRolloutsFakeRobot(directory+'/update00003')

    oneUpdate(directory,plot_results)

