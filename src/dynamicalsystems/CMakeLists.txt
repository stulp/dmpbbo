file(GLOB SOURCES *.cpp) 
add_library(dynamicalsystems ${SHARED_OR_STATIC} ${SOURCES})

install(TARGETS dynamicalsystems DESTINATION lib)
file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION include/dynamicalsystems/)

add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

