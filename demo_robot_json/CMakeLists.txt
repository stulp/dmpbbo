add_executable(robotExecuteTrajectory robotExecuteTrajectory.cpp runSimulationThrowBall.cpp)
target_link_libraries(robotExecuteTrajectory dmp functionapproximators dynamicalsystems eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS robotExecuteTrajectory DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(robotExecuteDmp robotExecuteDmp.cpp runSimulationThrowBall.cpp)
target_link_libraries(robotExecuteDmp dmp functionapproximators dynamicalsystems eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS robotExecuteDmp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
