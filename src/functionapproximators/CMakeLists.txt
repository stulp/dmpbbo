################################################################################
file(GLOB SOURCES leastSquares.cpp BasisFunction.cpp Parameterizable.cpp MetaParameters.cpp ModelParameters.cpp FunctionApproximator.cpp UnifiedModel.cpp *LWR.cpp *RBFN.cpp FunctionApproximatorFactory.cpp) 
if(LWPR_FOUND)
  if (BUILD_SHARED_LIBS) # LWPR doesn't work with shared libs
    message(WARNING  "LWPR cannot be used with .so libraries.")
  else()
    file(GLOB SOURCES ${SOURCES} *LWPR.cpp) 
  endif()
endif()

################################################################################
add_library(functionapproximators ${SHARED_OR_STATIC} ${SOURCES})
install(TARGETS functionapproximators DESTINATION ${LIB_INSTALL_DIR})

file(GLOB HEADERS *.hpp)
if(LWPR_FOUND)
  if (BUILD_SHARED_LIBS) # LWPR doesn't work with shared libs
    message(WARNING  "LWPR cannot be used with .so libraries.")
  else()
    file(GLOB HEADERS ${HEADERS} *LWPR.hpp) 
  endif()
endif()
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/functionapproximators)

#add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

