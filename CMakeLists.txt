# States that CMake required version must be greater than 2.6
cmake_minimum_required(VERSION 2.6)
# Project name is not mandatory, but you should use it
project(OO_DMP_BBO)
###############################################################################
# General settings

# Some settings related to building/installing libraries and header files
# Set this to STATIC if you want to build static libraries instead.
option(BUILD_SHARED_LIBS "Enable this option to build shared libraries" OFF)

if(BUILD_SHARED_LIBS)
set(SHARED_OR_STATIC "SHARED")
else()
set(SHARED_OR_STATIC "STATIC")
endif()


set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)


# Never build inside the source tree
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)



# Some flags for CXX
set(CMAKE_CXX_FLAGS "-Wall -std=c++0x -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Define macros to check for real time requirements
# To activate the EIGEN_MALLOC asserts use: cmake .. -DREALTIME_CHECKS=1 -DCMAKE_BUILD_TYPE=Debug
if(REALTIME_CHECKS)
   add_definitions(-DREALTIME_CHECKS)
endif()


# Appends the cmake/modules path inside the MAKE_MODULE_PATH variable which stores the
# directories of additional CMake modules (ie. MacroOutOfSourceBuild.cmake):
set(CMAKE_MODULE_PATH ${oo_dmp_bbo_SOURCE_DIR}/cmake/modules "${CMAKE_SOURCE_DIR}/src/functionapproximators/" ${CMAKE_MODULE_PATH})

include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${CMAKE_SOURCE_DIR}/lib)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "Comment" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

###############################################################################
# See if LWPR is installed. 
find_package(LWPR)
if(LWPR_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LWPR")
endif()

###############################################################################
# Find boost packages
find_package( Boost 1.34 COMPONENTS filesystem system serialization REQUIRED)
link_directories ( ${Boost_LIBRARY_DIRS} )
include_directories ( ${Boost_INCLUDE_DIRS} )

###############################################################################
# Things to compile are in the src/ directory
add_subdirectory(src)

###############################################################################
# Generate doxygen documentation from CMake
# http://www.bluequartz.net/projects/EIM_Segmentation/SoftwareDocumentation/html/usewithcmakeproject.html
option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" ON)
if(BUILD_DOCUMENTATION)
  FIND_PACKAGE(Doxygen)
  if (NOT DOXYGEN_FOUND)
    message(WARNING 
      "Doxygen is needed to build the documentation. Please install it correctly")
  endif()
  #-- Configure the Template Doxyfile for our specific project
  configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in 
                 ${PROJECT_BINARY_DIR}/Doxyfile  @ONLY IMMEDIATE)
  #-- Add a custom target to run Doxygen when ever the project is built
  add_custom_target (Docs  
  	COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
  	SOURCES ${PROJECT_BINARY_DIR}/Doxyfile)
  # IF you do NOT want the documentation to be generated EVERY time you build the project
  # then leave out the 'ALL' keyword from the above command.
endif()

