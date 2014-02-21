file(GLOB SOURCES *.cpp updaters/*.cpp) 
add_library(bbo ${SHARED_OR_STATIC} ${SOURCES})

install(TARGETS bbo DESTINATION ${LIB_INSTALL_DIR})
file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/bbo)

add_subdirectory(updaters)

add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
	add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

