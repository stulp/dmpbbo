file(GLOB HEADERS *.hpp *.tpp)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/dmpbbo_io)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
   add_subdirectory(tests)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug) 

