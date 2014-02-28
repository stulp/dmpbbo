file(GLOB SOURCES *.cpp) 
add_library(dmp_bbo ${SHARED_OR_STATIC} ${SOURCES})
install(TARGETS dmp_bbo DESTINATION ${LIB_INSTALL_DIR})

file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/dmp_bbo)

#target_link_libraries(dmp_bbo bbo)

add_subdirectory(tasks)
add_subdirectory(demos)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)
