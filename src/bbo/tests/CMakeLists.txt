add_executable(testDistributionGaussian testDistributionGaussian.cpp)
target_link_libraries(testDistributionGaussian bbo ${Boost_LIBRARIES})
install(TARGETS testDistributionGaussian DESTINATION bin_test)

add_executable(testUpdaters testUpdaters.cpp)
target_link_libraries(testUpdaters bbo ${Boost_LIBRARIES})
install(TARGETS testUpdaters DESTINATION bin_test)
