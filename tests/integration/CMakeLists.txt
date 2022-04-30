add_executable(testDynamicalSystems testDynamicalSystems.cpp)
target_link_libraries(testDynamicalSystems dynamicalsystems ${Boost_LIBRARIES})
install(TARGETS testDynamicalSystems DESTINATION bin)

add_executable(testFunctionApproximators testFunctionApproximators.cpp)
target_link_libraries(testFunctionApproximators functionapproximators ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximators DESTINATION bin)

add_executable(testDmp testDmp.cpp)
target_link_libraries(testDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES})
install(TARGETS testDmp DESTINATION bin)
