CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#set(DMPBBO_ROOT "/home/clement/Documents/Boulot/INRIA_FLOWERS/CODE/dmpbbo") 




FIND_PACKAGE(PythonLibs REQUIRED)

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        SET(dirlist ${dirlist} ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


#FIND_PACKAGE(Boost)
#find_package( Boost COMPONENTS filesystem system serialization REQUIRED)
#message("include : " ${Boost_INCLUDE_DIRS})
#message("libs : " ${Boost_LIBRARIES})

#IF(Boost_FOUND)

  #set( INCLUDE ${INCLUDE} ${Boost_INCLUDE_DIRS})
  #set( LIBS ${LIBS} ${Boost_LIBRARIES})

  ##INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIRS}")
  #SET(Boost_USE_STATIC_LIBS OFF)
  #SET(Boost_USE_MULTITHREADED ON)
  #SET(Boost_USE_STATIC_RUNTIME OFF)
  #FIND_PACKAGE(Boost COMPONENTS python)

  #ADD_LIBRARY(dmpbbo SHARED pydmpbbo.cpp main-python.cpp)
  #TARGET_LINK_LIBRARIES(dmpbbo ${Boost_LIBRARIES})
#ELSEIF(NOT Boost_FOUND)
  #MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
#ENDIF()

#set(func_appr_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE} CACHE PATH "functionapproximators include location")
#set(func_appr_LIBRARIES ${LIBS} CACHE PATH "functionapproximators libraries")
#set(func_appr_FOUND TRUE CACHE BOOL "functionapproximators Found!")

#set(func_appr_INCLUDE_DIRS ${INCLUDE_INSTALL_DIR}/functionapproximators)
#set(func_appr_LIBRARIES ${LIB_INSTALL_DIR}/libfunctionapproximators.so)


set(lib_dmpbbo_io "dmpbbo_io")
set(lib_dynamicalsystems "dynamicalsystems")

SUBDIRLIST(dmpbbo_DIRS ${CMAKE_SOURCE_DIR}/src)
FOREACH(subdir ${dmpbbo_DIRS})
    set(dir_str "${subdir}")
    IF(NOT (dir_str STREQUAL lib_dmpbbo_io OR dir_str STREQUAL lib_dynamicalsystems) )
        set(dmpbbo_libs ${dmpbbo_libs} "${CMAKE_BINARY_DIR}/src/${subdir}/lib${subdir}.so")
    ENDIF()
ENDFOREACH()

#message("__ : " ${CMAKE_SOURCE_DIR})

#file(GLOB dmpbbo_io_SOURCES ${CMAKE_SOURCE_DIR}/src/dmpbbo_io/*.cpp) 
#message("__ : " ${dmpbbo_io_SOURCES})
#add_library(dmpbbo_io ${SHARED_OR_STATIC} ${dmpbbo_io_SOURCES})

include_directories(${CMAKE_SOURCE_DIR}/src)
#link_directories(${CMAKE_SOURCE_DIR}/lib)

FIND_PACKAGE(Boost COMPONENTS python serialization filesystem system)
ADD_LIBRARY(dmpbbo SHARED pydmpbbo.cpp main-python.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#find_package(Eigen3 REQUIRED)
#include_directories(EIGEN3_INCLUDE_DIR)

TARGET_LINK_LIBRARIES(dmpbbo dmp dynamicalsystems functionapproximators ${LWPR_LIBRARIES}  ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
#message("include : " ${func_appr_INCLUDE_DIRS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") 
INCLUDE_DIRECTORIES("/usr/include/python2.7/" "/usr/lib/python2.7/dist-packages/numpy/" "/usr/lib/python2.7/dist-packages/numpy/core/include/")
#include_directories(${CMAKE_SOURCE_DIR}/src)
#link_directories(${CMAKE_SOURCE_DIR}/lib)
SET_TARGET_PROPERTIES(dmpbbo PROPERTIES OUTPUT_NAME "_dmpbbo" PREFIX "" SUFFIX ".so")


#include_directories(${INCLUDE})
#target_link_libraries(dmpbbo ${LIBS})
