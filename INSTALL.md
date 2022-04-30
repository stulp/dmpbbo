# Python version of dmpbbo

## Installing required/optional packages for debian users 

Tested under Ubuntu 20.04.

dmpbbo requires python3 (mainly due to the way packages are imported), numpy (http://www.numpy.org/) and matplotlib (https://matplotlib.org/), plus some latex packages for the plotting.

`sudo apt-get install python3-numpy python3-tk python3-matplotlib texlive texlive-latex-extra`

# C++ version of dmpbbo

## Installing required/optional packages for debian users 

Tested under Ubuntu 20.04

### Automatic installation

To automatically install the packages (required and optional) run:

`./install_dependencies.sh`

## Building dmpbbo 

Please make sure you have installed all the required packages before trying to compile. 
We assume that you use a compiler that is compatible with the C++11 standard.

There is a simple hand-coded Makefile in the root, which uses cmake to compile the C++ code. To compile optimized libraries/binaries do `make build`. To have a debug version (not optimized, contains debug symbols, contains real-time checks) fo `make build_debug`.

A local install in the dmpbbo root is achieved by `make install` or `make install_debug`. Either will overwrite the output of the other in `dmpbbo/lib/` and `dmpbbo/bin/`.
  
<a name="manual_installation"></a>
## Manual installation of required/optional packages

### cmake (required)
 http://www.cmake.org/cmake/resources/software.html	
  
### boost (required modules: filesystem, system, serialization. version >=1.34) 
  http://www.boost.org/doc/libs/1_55_0/more/getting_started/
  http://www.boost.org/doc/libs/1_55_0/libs/filesystem/doc/
  http://www.boost.org/doc/libs/1_55_0/libs/system/doc/
  http://www.boost.org/doc/libs/1_55_0/libs/serialization/doc/ 
  	
###  Doxygen (optional, version >1.7.5, for the \cite command)
  http://www.doxygen.nl/download.html

### Locally Weighted Projection Regression (LWPR) (optional)
  
To use the Locally Weighted Projection Regression algorithm as a  function approximator, install the following software: http://sourceforge.net/projects/lwpr/
Our build system assumes the resulting headers/libraries are installed in `/usr/local/include` and `/usr/local/lib` respectively. If they are in a  non-standard location (because you don't want to 'make install' or such)  please modify the following lines in `src/functionapproximators/FindLWPR.cmake` accordingly: 

+ `find_path(LWPR_INCLUDE_DIR lwpr.hh HINTS /usr/local/include/)`

+ ` find_library(LWPR_LIBRARY liblwpr.a HINTS /usr/local/lib/)`

