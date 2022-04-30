# Building the C++ version of dmpbbo

After having installed the required packages (see below), the simple hand-coded Makefile in the root is used to compile the C++ code. To compile optimized libraries/binaries do `make build`. To have a debug version (not optimized, contains debug symbols, contains real-time checks) do `make build_debug`.

A local install in the dmpbbo root is achieved by `make install` or `make install_debug`. Either will overwrite the output of the other one in `dmpbbo/lib/` and `dmpbbo/bin/`.

# Installing dependencies

To automatically install the packages (required and optional) run the following (Tested under Ubuntu 20.04):

`./install_dependencies.sh`

You can also manually install these dependencies. For instructions see below.

  
<a name="manual_installation"></a>
## Manual installation of required/optional packages for C++

### cmake (required)
 http://www.cmake.org/cmake/resources/software.html	
  
### boost (required modules: filesystem, system. version >=1.34) 
  http://www.boost.org/doc/libs/1_55_0/more/getting_started/
  http://www.boost.org/doc/libs/1_55_0/libs/filesystem/doc/
  http://www.boost.org/doc/libs/1_55_0/libs/system/doc/
  	
###  Doxygen (optional, version >1.7.5, for the \cite command)
  http://www.doxygen.nl/download.html