# This file is part of DmpBbo, a set of libraries and programs for the 
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
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

import os, sys
import numpy as np

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from functionapproximators.Parameterizable import Parameterizable

class FunctionApproximator(Parameterizable):
    
    def __init__(self):
        self._model_params = {}
        self._meta_params = {}
        self._selected_values_labels = []
        
    def train(self,inputs,targets):
        raise NotImplementedError('subclasses must override train()!')

    def predict(self,inputs):
        raise NotImplementedError('subclasses must override predict()!')
        
    def isTrained(self):
        raise NotImplementedError('subclasses must override isTrained()!')


    def setSelectedParameters(selected_values_labels):
        self._selected_values_labels = []      
        for label in selected_values_labels:
            if label in _model_params.keys():
                self._selected_values_labels.append(label)
            else:
                print(label+" not in ["+', '.join(_model_params.keys())+']: Ignoring')
                
    def getParameterVectorSelected(self):
        if self.isTrained():
            values = []
            for label in self._selected_values_labels:
                values.extend(self._model_params[label].flatten())
            return np.asarray(values)
        else:
            warning('FunctionApproximator is not trained.')
            return []
            
    def setParameterVectorSelected(self,values):
        if self.isTrained():
            if len(values)!=self.getParameterVectorSelectedSize():
                raise ValueError(f'values ({len(values)}) should have same size as size of selected parameters vector ({self.getParameterVectorSelectedSize()})')
            offset = 0
            for label in self._selected_values_labels:
                expected_shape = self._model_params[label].shape
                cur_n_values = np.prod(expected_shape)
                cur_values = values[offset:offset+cur_n_values]
                self._model_params[label] = np.reshape(cur_values,expected_shape)
                offset += cur_n_values
        else:
            warning('FunctionApproximator is not trained.')
            
    def getParameterVectorSelectedSize(self):
        size = 0
        for label in self._selected_values_labels:
            size += len(self._model_params[label].flatten())
        return size
