file(GLOB SOURCES *.cpp) 
add_library(dmp ${SHARED_OR_STATIC} ${SOURCES})

install(TARGETS dmp DESTINATION ${LIB_INSTALL_DIR})
file(GLOB HEADERS *.h *.hpp)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/dmp)

add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)
