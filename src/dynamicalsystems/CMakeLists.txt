file(GLOB HEADERS *.hpp)
file(GLOB SOURCES *.cpp) 

add_library(dynamicalsystems ${SHARED_OR_STATIC} ${SOURCES})

install(TARGETS dynamicalsystems DESTINATION ${LIB_INSTALL_DIR})
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/dynamicalsystems/)
