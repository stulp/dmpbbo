################################################################################
file(GLOB SOURCES BasisFunction.cpp Parameterizable.cpp ModelParameters.cpp FunctionApproximator.cpp *LWR.cpp *RBFN.cpp FunctionApproximatorFactory.cpp) 

################################################################################
add_library(functionapproximators ${SHARED_OR_STATIC} ${SOURCES})
install(TARGETS functionapproximators DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/functionapproximators)

#add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

