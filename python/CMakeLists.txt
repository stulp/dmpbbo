CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(PythonLibs REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)

FIND_PACKAGE(Boost COMPONENTS python serialization filesystem system)
ADD_LIBRARY(dmpbbo SHARED pydmpbbo.cpp main-python.cpp boostpython_eigen_conversion.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(dmpbbo dmp bbo dynamicalsystems functionapproximators ${LWPR_LIBRARIES}  ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

INCLUDE_DIRECTORIES("/usr/include/python2.7/" "/usr/lib/python2.7/dist-packages/numpy/" "/usr/lib/python2.7/dist-packages/numpy/core/include/")

SET_TARGET_PROPERTIES(dmpbbo PROPERTIES OUTPUT_NAME "_dmpbbo" PREFIX "" SUFFIX ".so")
add_executable(pydmpbbo test.cpp pydmpbbo.cpp boostpython_eigen_conversion.cpp)
TARGET_LINK_LIBRARIES(pydmpbbo dmp bbo dynamicalsystems functionapproximators ${LWPR_LIBRARIES}  ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

#target_link_libraries(dmpbbo ${LIBS})
