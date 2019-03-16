message("${LWPR_LIBRARIES}")

add_executable(demoDmp demoDmp.cpp)
target_link_libraries(demoDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoDmpChangeGoal demoDmpChangeGoal.cpp)
target_link_libraries(demoDmpChangeGoal dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpChangeGoal DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoDmpContextual demoDmpContextual.cpp)
target_link_libraries(demoDmpContextual dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpContextual DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoDmpContextualGoal demoDmpContextualGoal.cpp)
target_link_libraries(demoDmpContextualGoal dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpContextualGoal DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoDmpTrainFromTrajectoryFile demoDmpTrainFromTrajectoryFile.cpp)
target_link_libraries(demoDmpTrainFromTrajectoryFile dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpTrainFromTrajectoryFile DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoDmpWithGainSchedules demoDmpWithGainSchedules.cpp)
target_link_libraries(demoDmpWithGainSchedules dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS demoDmpWithGainSchedules DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

