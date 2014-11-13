################################################################################
file(GLOB SOURCES BasisFunction.cpp Parameterizable.cpp MetaParameters.cpp ModelParameters.cpp FunctionApproximator.cpp UnifiedModel.cpp *LWR.cpp *IRFRLS.cpp *GMR.cpp *RBFN.cpp *GPR.cpp getFunctionApproximatorByName.cpp) 
if(LWPR_FOUND)
  file(GLOB SOURCES ${SOURCES} *LWPR.cpp) 
endif()

################################################################################
add_library(functionapproximators ${SHARED_OR_STATIC} ${SOURCES})
install(TARGETS functionapproximators DESTINATION ${LIB_INSTALL_DIR})

file(GLOB HEADERS *.hpp)
if(LWPR_FOUND)
  file(GLOB HEADERS ${HEADERS} *LWPR.hpp) 
endif()
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/functionapproximators)

add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

