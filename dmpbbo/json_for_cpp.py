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
""" Module with functions to save to/load from json. """

import json
from json import JSONEncoder

import jsonpickle
import numpy as np

# Using jsonpickle to generate JSON that can be read by nlohmann::json was difficult.
#
# Standard jsonpickle replaces duplicate objects with their ids to save space. This
# is challenging to parse in C++.
#
# With make_refs=False, there are no refs, but it outputs np.array weirdly.
#
# The simplest solution was to make a small DmpBboJSONEncoder for json, rather than
# using jsonpickle


class DmpBboJSONEncoder(JSONEncoder):
    """ Custom encoder for converting objects to JSON.
    """

    def default(self, obj):
        """ Override the default implementation for encoding.

        @param obj:  Object to encode
        @return: JSON representation
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            # Add class to the json output. Necessary for reading it in C++
            d = {"class": obj.__class__.__name__}
            d.update(obj.__dict__)
            return d
        return json.JSONEncoder.default(self, obj)


def savejson_for_cpp(filename, obj):
    """ Save an object to a file that can be read by the C++ implementation of dmpbbo.

    @param filename: Name of the file to save to.
    @param obj: Object to save.
    """
    # Save a simple JSON version that can be read by the C++ code.
    filename = str(filename)  # In case it is path
    j = json.dumps(obj, cls=DmpBboJSONEncoder, indent=2)
    with open(filename, "w") as out_file:
        out_file.write(j)


def savejson(filename, obj):
    """ Save an object to a json file with jsonpickle.

    @param filename: Name of the file to save to.
    @param obj: Object to save.
    """
    # Save to standard jsonpickle file
    j = jsonpickle.encode(obj)
    with open(filename, "w") as out_file:
        out_file.write(j)


def loadjson(filename):
    """ Load an object from a file with json with jsonpickle.

    @param filename: Name of the file to load from.
    @return Object that was loaded.
    """
    # Load from standard jsonpickle file
    with open(filename, "r") as in_file:
        j = in_file.read()
    obj = jsonpickle.decode(j)
    return obj
