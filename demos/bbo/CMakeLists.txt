add_executable(demoOptimization demoOptimization.cpp)
target_link_libraries(demoOptimization bbo ${Boost_LIBRARIES})
install(TARGETS demoOptimization DESTINATION bin)



