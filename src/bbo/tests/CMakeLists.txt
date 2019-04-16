add_executable(testDistributionGaussian testDistributionGaussian.cpp)
target_link_libraries(testDistributionGaussian bbo ${Boost_LIBRARIES})

add_executable(testUpdaters testUpdaters.cpp)
target_link_libraries(testUpdaters bbo ${Boost_LIBRARIES})