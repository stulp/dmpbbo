# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2022 Freek Stulp
# 
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os, sys

# Include scripts for plotting
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from to_jsonpickle import *

from dynamicalsystems.ExponentialSystem import *
from dynamicalsystems.SigmoidSystem import *
from dynamicalsystems.SpringDamperSystem import *
from dynamicalsystems.TimeSystem import *

def save_jsonpickle(obj,filename):
    
    s = to_jsonpickle(obj)
    
    # Save to file
    with open(filename, "w") as text_file:
        text_file.write(s)

if __name__=='__main__':
    """Run some training sessions and plot results."""

    for n_dims in [1,2]:

        # ExponentialSystem
        tau = 0.6 # Time constant
        if (n_dims==1):
            y_init = np.array([0.5])
            y_attr = np.array([0.8])
        else:
            y_init = np.array([0.5, 1.0])
            y_attr = np.array([0.8, 0.1])
            
        alpha = 6.0 # Decay factor
        dyn_system = ExponentialSystem(tau, y_init, y_attr, alpha)
        save_jsonpickle(dyn_system,f"ExponentialSystem_{n_dims}D.json")
              
        # SigmoidSystem
        max_rate = -20
        inflection_point = tau*0.8
        dyn_system = SigmoidSystem(tau, y_init, max_rate, inflection_point)
        save_jsonpickle(dyn_system,f"SigmoidSystem_{n_dims}D.json")
    
        # SpringDamperSystem
        alpha = 12.0
        dyn_system = SpringDamperSystem(tau, y_init, y_attr, alpha)
        save_jsonpickle(dyn_system,f"SpringDamperSystem_{n_dims}D.json")
    
    # TimeSystem
    dyn_system = TimeSystem(tau)
    save_jsonpickle(dyn_system,f"TimeSystem.json")

    # TimeSystem (but counting down instead of up)
    count_down = True
    dyn_system = TimeSystem(tau,count_down,"TimeSystemCountDown")
    save_jsonpickle(dyn_system,f"TimeSystemCountDown.json")


