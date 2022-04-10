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
import pprint

import json

import jsonpickle

from jsonpickle import handlers
from jsonpickle import Pickler
from jsonpickle import Unpickler

import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
handler = jsonpickle.ext.numpy.NumpyNDArrayHandlerView(size_threshold=None)
handlers.registry.unregister(np.ndarray)
handlers.registry.register(np.ndarray, handler, base=True)
    
from collections import MutableMapping
from contextlib import suppress

def delete_keys_from_dict(dictionary, keys):
    #https://stackoverflow.com/questions/3405715/elegant-way-to-remove-fields-from-nested-dictionaries
    #d = {'c':[{'base':'yo','d':'e'},{'f':'g'}], 'a':'b'}
    #delete_keys_from_dict(d,['base'])
    #print(d)
    if isinstance(dictionary, list):
        for d in dictionary:
            delete_keys_from_dict(d,keys)
    elif isinstance(dictionary, MutableMapping):
        for key in keys:
            with suppress(KeyError):
                del dictionary[key]
        for value in dictionary.values():
            delete_keys_from_dict(value, keys)
            

def from_jsonpickle(json_string):
    j = json.loads(json_string)
    unpickler = Unpickler()
    return unpickler._restore(j)
    
def to_jsonpickle(obj):
    
    pickler = Pickler()
    # jsonpickle.encode or picker.flatten have issues with refs
    # Calling the underscore function _flatten_obj works however
    json_dict = pickler._flatten_obj(obj)
    
    # Delete some keys that are not necessary for unpickling
    # This leads to smaller files, and avoids py\id issues and parse 
    # issues with shape.
    delete_keys_from_dict(json_dict,['base','shape','strides'])

    # Pretty-print the format, and make minor replacements that allow
    # nlohmann::json to read the string (and are compatible with jsonpickle
    # also)
    s = pprint.pformat(json_dict,compact=True)
    s = s.replace("'",'"')
    s = s.replace("'",'"')
    s = s.replace("False","0")
    s = s.replace("True","1")
    
    return s
        
    # This did not work. 
    # With make_refs=True, jsonpickle makes refs rather than pickling the
    # object. This is challenging to parse in C++.
    # With make_refs=False, there are no refs, but 
    #jp = jsonpickle.encode(obj,make_refs=False)
    #j = json.dumps(json.loads(jp), indent=2, sort_keys=False)

    # This could work, but has the disadvantage that known classes can be
    # converted. One would  thus have to write an encoder for each class in
    # DmpBbo
    # class DmpBboEncoder(json.JSONEncoder):
    #    def default(self, obj):
    #        if isinstance(obj, np.ndarray):
    #            return obj.tolist()
    #        if isinstance(obj, FunctionApproximatorRBFN):
    #            return "yo"
    #        if isinstance(obj, FunctionApproximatorLWR):
    #            return "yo"
    #        return json.JSONEncoder.default(self, obj)
    #
    # j = json.dumps(obj.__dict__, cls=DmpBboEncoder)

    
