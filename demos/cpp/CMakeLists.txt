add_executable(demoDynamicalSystems demoDynamicalSystems.cpp)
target_link_libraries(demoDynamicalSystems dynamicalsystems ${Boost_LIBRARIES})
install(TARGETS demoDynamicalSystems DESTINATION bin)

add_executable(demoFunctionApproximators demoFunctionApproximators.cpp)
target_link_libraries(demoFunctionApproximators functionapproximators ${Boost_LIBRARIES})
install(TARGETS demoFunctionApproximators DESTINATION bin)

add_executable(demoTrajectory demoTrajectory.cpp)
target_link_libraries(demoTrajectory dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES})
install(TARGETS demoTrajectory DESTINATION bin)

add_executable(demoDmp demoDmp.cpp)
target_link_libraries(demoDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES})
install(TARGETS demoDmp DESTINATION bin)