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


## \file demoDmpTrainFromTrajectoryFile.py
## \author Freek Stulp
## \brief  A simple wrapper around demoDmpTrainFromTrajectoryFile.cpp, just for consistency
## 
## \ingroup Demos
## \ingroup Dmps

import numpy
import matplotlib.pyplot as plt
import os, sys

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from executeBinary import executeBinary

if __name__=='__main__':

    # Call the executable with the directory to which results should be written
    executable = "../../bin/demoDmpTrainFromTrajectoryFile"
    input_txt_file = "trajectory.txt"
    output_xml_file = "/tmp/dmp.xml"
    executeBinary(executable, input_txt_file+" "+output_xml_file,True)
