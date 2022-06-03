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
""" Module for the Parameterizable class. """

from abc import ABC, abstractmethod


class Parameterizable(ABC):
    """ Interface for providing access to a model's parameters as a vector.

    Different function approximators have different types of model parameters. For instance,
    LWR has the centers and widths of basis functions, along with the slopes of each line segment.
    get_param_vector provides a means to access these parameters as one vector.

    Which parameters are returned can be set with set_selected_param_names, e.g.
    set_selected_param_names(["slopes","offsets"])

    This may be useful for instance when optimizing the model parameters with black-box
    optimization, which is agnostic about the semantics of the model parameters.
    """

    @abstractmethod
    def set_selected_param_names(self, names):
        """ Set the selected parameters.

        @param names: Name of the parameter to select.
        """
        pass

    @abstractmethod
    def get_param_vector(self):
        """Get a vector containing the values of the selected parameters."""
        pass

    @abstractmethod
    def set_param_vector(self, values):
        """Set a vector containing the values of the selected parameters."""
        pass

    @abstractmethod
    def get_param_vector_size(self):
        """Get the size of the vector containing the values of the selected parameters."""
        pass
