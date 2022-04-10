add_executable(testJsonFunctionApproximators testJsonFunctionApproximators.cpp)
target_link_libraries(testJsonFunctionApproximators functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorCloning testFunctionApproximatorCloning.cpp)
target_link_libraries(testFunctionApproximatorCloning functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorRealtime testFunctionApproximatorRealtime.cpp)
target_link_libraries(testFunctionApproximatorRealtime functionapproximators eigen ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(testModelParameterValues testModelParameterValues.cpp)
target_link_libraries(testModelParameterValues functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})


################################################################################
# LWR
add_executable(testMetaParametersLWR testMetaParametersLWR.cpp)
target_link_libraries(testMetaParametersLWR functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
