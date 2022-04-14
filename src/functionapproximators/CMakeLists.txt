file(GLOB HEADERS *.hpp)
file(GLOB SOURCES *.cpp) 

add_library(functionapproximators ${SHARED_OR_STATIC} ${SOURCES})

install(TARGETS functionapproximators DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/functionapproximators)