add_executable(testTaskViapoint testTaskViapoint.cpp)
target_link_libraries( testTaskViapoint taskviapoint dmp dynamicalsystems functionapproximators bbo ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
#install(TARGETS testTaskViapoint DESTINATION bin_test)
