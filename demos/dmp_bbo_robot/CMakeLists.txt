add_executable(trainDmpFromTrajectoryFile trainDmpFromTrajectoryFile.cpp)
target_link_libraries(trainDmpFromTrajectoryFile dmp functionapproximators dynamicalsystems ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS trainDmpFromTrajectoryFile DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(executeDmp executeDmp.cpp)
target_link_libraries(executeDmp dmp functionapproximators dynamicalsystems ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS executeDmp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})