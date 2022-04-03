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

import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import pprint

# Include scripts for plotting
lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from functionapproximators.FunctionApproximatorLWR import *
from functionapproximators.FunctionApproximatorRBFN import *


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.squeeze().tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def toJson(obj):
    new_dict = {"class": obj.__class__.__name__}
    for key, val in obj.__dict__.items():
        #new_key = key[1:] if key[0]=='_' else key
        new_dict[key] = val 
    return json.dumps(new_dict,cls=NumpyArrayEncoder)

if __name__=='__main__':
    """Run some training sessions and plot results."""

    np.set_printoptions(precision=4)

    # Generate training data 
    n_samples_per_dim = 25
    #n_samples_per_dim = [11,9] # Does not work yet; kept for future debugging.
    n_dims = 1 if np.isscalar(n_samples_per_dim) else len(n_samples_per_dim)
    if n_dims==1:
        inputs = np.linspace(0.0, 2.0,n_samples_per_dim)
        targets = 3*np.exp(-inputs)*np.sin(2*np.square(inputs))
    else:
        n_samples = np.prod(n_samples_per_dim)
        # Here comes naive inefficient implementation...
        x1s = np.linspace(-2.0, 2.0,n_samples_per_dim[0])
        x2s = np.linspace(-2.0, 2.0,n_samples_per_dim[1])
        inputs = np.zeros((n_samples,n_dims))
        targets = np.zeros(n_samples)
        ii = 0
        for x1 in x1s:
            for x2 in x2s:
                inputs[ii,0] = x1
                inputs[ii,1] = x2
                targets[ii] = 2.5*x1*np.exp(-np.square(x1)-np.square(x2))
                ii += 1
               
    fa_names = ["RBFN","LWR"]
    for fa_index in range(len(fa_names)):
        fa_name = fa_names[fa_index]
        
        # Initialize function approximator
        if fa_name=="LWR":
            intersection = 0.5;
            n_rfs = 9;
            fa = FunctionApproximatorLWR(n_rfs,intersection)
        else:
            intersection = 0.7;
            n_rfs = 9;
            fa = FunctionApproximatorRBFN(n_rfs,intersection)
        
        # Train function approximator with data
        fa.train(inputs,targets)
        
        #jsonpickle.set_encoder_options('json', indent=4)
        #print(jsonpickle.encode(fa,cls=NumpyArrayEncoder))
        j = json.dumps(json.loads(jsonpickle.encode(fa)), indent=4, sort_keys=True)
        with open(fa_name+".json", "w") as text_file:
            text_file.write(j)

