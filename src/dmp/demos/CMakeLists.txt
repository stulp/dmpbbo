message("${LWPR_LIBRARIES}")

add_executable(demoDmp demoDmp.cpp)
target_link_libraries(demoDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmp DESTINATION bin)

add_executable(demoDmpContextual demoDmpContextual.cpp)
target_link_libraries(demoDmpContextual dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpContextual DESTINATION bin)

