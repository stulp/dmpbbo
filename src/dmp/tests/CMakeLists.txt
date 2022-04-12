add_executable(testTrajectory testTrajectory.cpp)
target_link_libraries(testTrajectory dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testJsonDmp testJsonDmp.cpp)
target_link_libraries(testJsonDmp dmp dynamicalsystems functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})