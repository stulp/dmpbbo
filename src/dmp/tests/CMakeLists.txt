add_executable(testTrajectory testTrajectory.cpp)
target_link_libraries(testTrajectory dmp dynamicalsystems functionapproximators eigen ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testDmp testDmp.cpp testDynamicalSystemFunction.cpp)
target_link_libraries(testDmp dmp dynamicalsystems functionapproximators eigen ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testDmpSerialization testDmpSerialization.cpp)
target_link_libraries(testDmpSerialization dmp dynamicalsystems functionapproximators eigen ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testDmpModelParameterValues testDmpModelParameterValues.cpp)
target_link_libraries(testDmpModelParameterValues dmp dynamicalsystems functionapproximators eigen ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
