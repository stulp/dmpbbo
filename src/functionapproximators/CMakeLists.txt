################################################################################
file(GLOB SOURCES Parameterizable.cpp MetaParameters.cpp ModelParameters.cpp FunctionApproximator.cpp  *LWR.cpp *IRFRLS.cpp *GMR.cpp *RBFN.cpp *GPR.cpp) 
if(LWPR_FOUND)
  file(GLOB SOURCES ${SOURCES} *LWPR.cpp) 
endif()

################################################################################
add_library(functionapproximators ${SHARED_OR_STATIC} ${SOURCES})
install(TARGETS functionapproximators DESTINATION lib)

file(GLOB HEADERS *.hpp)
if(LWPR_FOUND)
  file(GLOB HEADERS ${HEADERS} *LWPR.hpp) 
endif()
install(FILES ${HEADERS} DESTINATION include/functionapproximators)

add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

