file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/eigen_realtime)

#IF(CMAKE_BUILD_TYPE MATCHES Debug)
#   add_subdirectory(tests)
#ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

