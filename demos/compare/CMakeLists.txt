add_executable(compareDynamicalSystems compareDynamicalSystems.cpp)
target_link_libraries(compareDynamicalSystems dynamicalsystems ${Boost_LIBRARIES})

#add_executable(demoFunctionApproximators demoFunctionApproximators.cpp)
#target_link_libraries(demoFunctionApproximators functionapproximators ${Boost_LIBRARIES})

#add_executable(demoTrajectory demoTrajectory.cpp)
#target_link_libraries(demoTrajectory dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES})

#add_executable(demoDmp demoDmp.cpp)
#target_link_libraries(demoDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES})