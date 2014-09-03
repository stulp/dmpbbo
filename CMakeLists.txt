# States that CMake required version must be greater than 2.6
if(APPLE)
# RPATH support for Dynamic libs
cmake_minimum_required(VERSION 2.8.12)
else()
cmake_minimum_required(VERSION 2.6)
endif()
# Project name is not mandatory, but you should use it
project(OO_DMP_BBO)
###############################################################################
# General settings

# Some settings related to building/installing libraries and header files
# Set this to STATIC if you want to build static libraries instead.
option(BUILD_SHARED_LIBS "Enable this option to build shared libraries" ON)

if(BUILD_SHARED_LIBS)
set(SHARED_OR_STATIC "SHARED")
else()
set(SHARED_OR_STATIC "STATIC")
endif()


#set(LIB_INSTALL_DIR /usr/local/lib)
#set(INCLUDE_INSTALL_DIR /usr/local/include)


# Never build inside the source tree
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)



# Some flags for CXX
if(APPLE)
  set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -stdlib=libc++")
  set(CMAKE_MACOSX_RPATH 1)
else()
  set(CMAKE_CXX_FLAGS "-Wall -std=c++0x")
endif(APPLE)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Appends the cmake/modules path inside the MAKE_MODULE_PATH variable which stores the
# directories of additional CMake modules (ie. MacroOutOfSourceBuild.cmake):
set(CMAKE_MODULE_PATH ${oo_dmp_bbo_SOURCE_DIR}/cmake/modules "${CMAKE_SOURCE_DIR}/src/functionapproximators/" ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules)

include_directories(${CMAKE_SOURCE_DIR}/src)
###link_directories(${CMAKE_SOURCE_DIR}/lib)

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
###link_directories ( ${Boost_LIBRARY_DIRS} )
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


###############################################################################
###############################################################################
###############################################################################
######## 			Debian 	packaging			#######
###############################################################################
###############################################################################
###############################################################################
###############################################################################
OPTION(ENABLE_PPA "Enable PPA creation" FALSE)
OPTION(UPLOAD_PPA "Enable PPA upload" FALSE)
OPTION(CHANGELOG_MESSAGE "Set the changelog message" OFF)

set(CPACK_DEBIAN_PACKAGE_NAME "dmpbbo")
SET(MAJOR_VERSION "0")
SET(MINOR_VERSION "9")
SET(PATCH_VERSION "4")
set(M3CORE_VERSION
  ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})
#### Package Generation
execute_process (
  COMMAND          /usr/bin/dpkg --print-architecture
  OUTPUT_VARIABLE  CPACK_DEBIAN_PACKAGE_ARCHITECTURE
  RESULT_VARIABLE  EXECUTE_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(EXECUTE_RESULT)
  message(STATUS "dpkg not found: No package generation.")
else()
IF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
INCLUDE(InstallRequiredSystemLibraries)
message(STATUS "Debian architecture: ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

SET(CPACK_GENERATOR "DEB")

SET(CPACK_PACKAGE_DESCRIPTION "Dynamic Movement Primitives and Black-Box Optimization C++ library.")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "This repository provides an implementation of dynamical systems, function approximators, dynamical movement primitives, and black-box optimization with evolution strategies, in particular the optimization of the parameters of dynamical movement primitives.")

SET(CPACK_PACKAGE_VENDOR "Ensta ParisTech, Inria")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Antoine Hoarau <hoarau.robotics@gmail.com>")
SET(CPACK_PACKAGE_CONTACT "Freek Stulp <stulp@ensta.fr>")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/stulp/dmpbbo")
SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
SET(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${MAJOR_VERSION}.${MINOR_VERSION}.${CPACK_PACKAGE_VERSION_PATCH}")

set(CPACK_STRIP_FILES TRUE)
set(PPA_DEBIAN_VERSION "0ubuntu1")
#dependencies for this service menu
set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS cmake bzr git pkg-config devscripts dh-make libeigen3-dev libboost-filesystem-dev libboost-system-dev libboost-serialization-dev doxygen graphviz)
set(CPACK_DEBIAN_PACKAGE_DEPENDS libeigen3-dev libboost-filesystem-dev libboost-system-dev libboost-serialization-dev graphviz)

set(DPUT_HOST hoarau-robotics)
set(CPACK_DEBIAN_UPDATE_CHANGELOG TRUE)
#SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
#SET(CPACK_DEBIAN_PACKAGE_SECTION "libraries")
 
#SET(CPACK_COMPONENTS_ALL Libraries ApplicationData)
INCLUDE(CPack)
if(ENABLE_PPA OR UPLOAD_PPA)
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "any") # can be build on any system
    include(UploadPPA)
endif()

ENDIF(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
endif()


