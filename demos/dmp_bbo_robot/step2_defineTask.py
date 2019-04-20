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
import matplotlib.pyplot as plt
import pickle
import inspect

lib_path = os.path.abspath('../../python')
sys.path.append(lib_path)

from TaskThrowBall import TaskThrowBall


if __name__=="__main__":
    
    output_task_file = None
    
    if (len(sys.argv)<2):
        print('Usage: '+sys.argv[0]+' <task file.p>')
        print('Example: python3 '+sys.argv[0]+' results/task.p')
        sys.exit()
        
    if (len(sys.argv)>1):
        output_task_file = sys.argv[1]

    x_goal = -0.70
    x_margin = 0.01
    y_floor = -0.3
    acceleration_weight = 0.001
    task = TaskThrowBall(x_goal,x_margin,y_floor,acceleration_weight)
    
    # Save the task instance itself
    print('  * Saving task to file "'+output_task_file+"'")
    pickle.dump(task, open(output_task_file, "wb" ))

    # Save the source code of the task for future reference
    #src_task = inspect.getsourcelines(task.__class__)
    #src_task = ' '.join(src_task[0])
    #src_task = src_task.replace("(Task)", "")    
    #filename = directory+'/the_task.py'
    #task_file = open(filename, "w")
    #task_file.write(src_task)
    #task_file.close()
