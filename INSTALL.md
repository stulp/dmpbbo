# Python version of dmpbbo

## Installing required/optional packages for debian users 

Tested under Ubuntu 16.04.

dmpbbo requires python3 (mainly due to the way packages are imported), numpy (http://www.numpy.org/) and matplotlib (https://matplotlib.org/), plus some latex packages for the plotting.

`sudo apt-get install python3-numpy python3-tk python3-matplotlib texlive texlive-latex-extra`

# C++ version of dmpbbo

## Installing required/optional packages for debian users 

Tested under Ubuntu 16.04

### Automatic installation

To automatically install the packages (required and optional) run:

`./install_dependencies.sh`

Otherwise, for manual installation see [below](#manual_installation).

### Required packages

`sudo apt-get install cmake libboost-filesystem-dev libboost-system-dev libboost-serialization-dev libeigen3-dev `

### Optional packages

`sudo apt-get install doxygen graphviz`

## Building dmpbbo 

Please make sure you have installed all the required packages before trying to compile. 
We assume that you use a compiler that is compatible with the C++11 standard.

To compile, do:
 `mkdir -p build_dir; cd build_dir; cmake .. -DCMAKE_BUILD_TYPE=Release; make; make Docs`

When running the code on a real robot, you want to do an optimized build without asserts, i.e. `-DCMAKE_BUILD_TYPE=Release`

To install libs in `/usr/local/lib`, headers in `/usr/local/include` and binaries in `dmpbbo/bin`:
 `sudo make install`

### Static vs. shared libraries

The default is to build static libraries (e.g. `libdmp.a`) and to statically link these libraries to the demo and test executables (the reason for this is the boost bug below).

Alternatively, you can build shared libraries by changing the CMakeLists.txt in this directory:
 `set(SHARED_OR_STATIC "SHARED") => set(SHARED_OR_STATIC "STATIC")`
Calling `sudo make install` will install the shared libraries  in `/usr/local/lib`. Make sure that `/usr/local/lib` is in your `LD_LIBRARY_PATH`, or  the binaries will not be able to dynamically link the required libraries.

If you use shared libraries, your program may crash with:

  +  pure virtual method called
  
  + terminate called without an active exception
  
This is a bug in `boost::serialization` for versions >=1.44, which can be fixed with this patch:

+ https://svn.boost.org/trac/boost/ticket/4842#comment:21

+ https://svn.boost.org/trac/boost/attachment/ticket/4842/void_cast.cpp.patch

Note that the crash happens towards the very end of the program, just before returning from main() when desctructors are called. So it does not affect functionality, but it is just very irritating that boost crashes and sometimes dumps a core (I use boost version 1.48, maybe the issue is fixed in later versions). If I had known things like this would happen, I would have probably used cereal (http://uscilab.github.io/cereal/) rather than `boost::serialization`... 

## Building dmpbbo (advanced build options for debugging purposes)

# Including debugging symbols and tests

To compile a version that has debugging symbols included, and compiles some tests also:
`mkdir -p build_dir_debug; cd build_dir_debug; cmake .. -DCMAKE_BUILD_TYPE=Debug; make VERBOSE=1`

# Debugging real-time code

Various function approximators have been optimized so that no dynamic allocations are made in real-time critical functions. The Eigen matrix library has functionality for checking whether dynamic allocations are made in certain blocks of code. 

To compile a version that has real-time checks built in:
  `mkdir -p build_dir_realtime; cd build_dir_realtime; cmake .. -DREALTIME_CHECKS=1 -DCMAKE_BUILD_TYPE=Debug; make VERBOSE=1`

Note that it doesn't make sense to set `-DREALTIME_CHECKS=1` without `-DCMAKE_BUILD_TYPE=Debug`, as asserts are only done in debug mode. 
  
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

