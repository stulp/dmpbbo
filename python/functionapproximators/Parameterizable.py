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

class Parameterizable:
    

    def getSelectableParameters(self):
        """Return the names of the parameters that can be selected.
        """
        raise NotImplementedError('subclasses must override getSelectableParameters()!')

    def getSelectableParametersRecommended(self):
        """Return the names of the parameters that recommended to be selected.
        """
        raise NotImplementedError('subclasses must override getSelectableParametersRecommended()!')
        
    def setSelectedParameters(selected_values_labels):
        """Set the selected parameters."""
        raise NotImplementedError('subclasses must override setSelectedParameters()!')
  
    def getParameterVectorSelected(self):
        """Get a vector containing the values of the selected parameters."""
        raise NotImplementedError('subclasses must override getParameterVectorSelected()!')
    
    def setParameterVectorSelected(self,values):
        """Set a vector containing the values of the selected parameters."""
        raise NotImplementedError('subclasses must override setParameterVectorSelected()!')
        
    def getParameterVectorSelectedSize(self):
        """Get the size of the vector containing the values of the selected parameters."""
        raise NotImplementedError('subclasses must override getParameterVectorSelectedSize()!')
        
        

