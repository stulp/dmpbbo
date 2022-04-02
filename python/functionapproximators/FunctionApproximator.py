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
import warnings

lib_path = os.path.abspath('../../../python/')
sys.path.append(lib_path)

from functionapproximators.Parameterizable import Parameterizable

class FunctionApproximator(Parameterizable):
    """Base class for all function approximators.

    See https://github.com/stulp/dmpbbo/blob/master/tutorial/functionapproximators.md
    """
    
    def __init__(self,meta_params):
        """Initialize a function approximator with meta- and optionally model-parameters
        
        Args:
           meta_parameters (dict): The meta-parameters for the training algorithm 
        """
        self._meta_params = meta_params
        self._model_params = None
        self._selected_param_labels = self.getSelectableParametersRecommended()
        
    def train(self,inputs,targets):
        """Train the function approximator with input and target examples.
        
        Args:
            inputs (numpy.ndarray): Input values of the training examples.
            targets (numpy.ndarray): Target values of the training examples.
        """
        raise NotImplementedError('subclasses must override train()!')

    def predict(self,inputs):
        """Query the function approximator to make a prediction.
        
        Args:
            inputs (numpy.ndarray): Input values of the query.
            
        Returns:
            numpy.ndarray: Predicted output values.
        """
        raise NotImplementedError('subclasses must override predict()!')
        
    def isTrained(self):
        """Determine whether the function approximator has already been trained with data or not.
        
        Returns:
            bool: True if the function approximator has already been trained, False otherwise.
        """
        raise NotImplementedError('subclasses must override isTrained()!')

    def setSelectedParameters(self,selected_param_labels):
        """Implements abstract function from the Parameterizable abstract class.
        """
        selectable_param_labels = self.getSelectableParameters()
        self._selected_param_labels = []
        if isinstance(selected_param_labels,str):
            # Make sure it is a list
            selected_param_labels = [selected_param_labels]
        for label in selected_param_labels:
            if not label in selectable_param_labels:
                warnings.warn(label+" not in ["+', '.join(selectable_param_labels)+']: Ignoring')
            else:
                self._selected_param_labels.append(label)
                
    def getParameterVectorSelected(self):
        """Implements abstract function from the Parameterizable abstract class.
        """
        if not self.isTrained():
            raise ValueError('FunctionApproximator is not trained.')
            
        values = []
        for label in self._selected_param_labels:
            values.extend(self._model_params[label].flatten())
        return np.asarray(values)
            
    def setParameterVectorSelected(self,values):
        """Implements abstract function from the Parameterizable abstract class.
        """
        if not self.isTrained():
            raise ValueError('FunctionApproximator is not trained.')
            
        if len(values)!=self.getParameterVectorSelectedSize():
            raise ValueError(f'values ({len(values)}) should have same size as size of selected parameters vector ({self.getParameterVectorSelectedSize()})')
            
        offset = 0
        for label in self._selected_param_labels:
            expected_shape = self._model_params[label].shape
            cur_n_values = np.prod(expected_shape)
            cur_values = values[offset:offset+cur_n_values]
            self._model_params[label] = np.reshape(cur_values,expected_shape)
            offset += cur_n_values
            
    def getParameterVectorSelectedSize(self):
        """Implements abstract function from the Parameterizable abstract class.
        """
        size = 0
        for label in self._selected_param_labels:
            if label in self._model_params:
                size += np.prod(self._model_params[label].shape)
        return size