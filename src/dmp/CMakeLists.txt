file(GLOB SOURCES *.cpp) 
add_library(dmp ${SHARED_OR_STATIC} ${SOURCES})

install(TARGETS dmp DESTINATION lib)
file(GLOB HEADERS *.h *.hpp)
install(FILES ${HEADERS} DESTINATION include/dmp)

add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)
