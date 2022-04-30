#!/bin/bash

sudo apt-get update -qq

##################
# For python
echo "Python"
sudo apt-get install -qq python3 python3-numpy python3-tk python3-matplotlib
# For testing
sudo apt-get install -qq python-pytest

##################
# For C++
echo "C++"
sudo apt-get install -qq cmake 
sudo apt-get install -qq gcc
sudo apt-get install -qq libboost-filesystem-dev libboost-system-dev 
sudo apt-get install -qq libeigen3-dev nlohmann-json3-dev 
# For documentation
echo "C++ documentation"
sudo apt-get install -qq doxygen graphviz texlive texlive-latex-extra

