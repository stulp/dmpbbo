add_executable(trainDmpFromTrajectoryFile trainDmpFromTrajectoryFile.cpp)
target_link_libraries(trainDmpFromTrajectoryFile dmp functionapproximators dynamicalsystems ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS trainDmpFromTrajectoryFile DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(performDmpRollouts performDmpRollouts.cpp)
target_link_libraries(performDmpRollouts dmp functionapproximators dynamicalsystems ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS performDmpRollouts DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})