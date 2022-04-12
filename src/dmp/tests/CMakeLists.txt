add_executable(testTrajectory testTrajectory.cpp)
target_link_libraries(testTrajectory dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testDmp testDmp.cpp)
target_link_libraries(testDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})