################################################################################
file(GLOB SOURCES eigen_json.cpp) 

add_library(eigen ${SHARED_OR_STATIC} ${SOURCES})
install(TARGETS eigen DESTINATION ${LIB_INSTALL_DIR})

file(GLOB HEADERS *.hpp)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/eigen)