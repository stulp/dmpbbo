add_executable(testFunctionApproximatorCloning testFunctionApproximatorCloning.cpp getFunctionApproximatorsVector.cpp)
target_link_libraries(testFunctionApproximatorCloning functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximatorCloning DESTINATION bin_test)

add_executable(testFunctionApproximatorTraining testFunctionApproximatorTraining.cpp getFunctionApproximatorsVector.cpp)
target_link_libraries(testFunctionApproximatorTraining functionapproximators ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximatorTraining DESTINATION bin_test)

add_executable(testFunctionApproximatorSerialization testFunctionApproximatorSerialization.cpp getFunctionApproximatorsVector.cpp)
target_link_libraries(testFunctionApproximatorSerialization functionapproximators ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximatorSerialization DESTINATION bin_test)

add_executable(testModelParameterValues testModelParameterValues.cpp)
target_link_libraries(testModelParameterValues functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testModelParameterValues  DESTINATION bin_test)


################################################################################
# LWR
add_executable(testMetaParametersLWR testMetaParametersLWR.cpp)
target_link_libraries(testMetaParametersLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testMetaParametersLWR  DESTINATION bin_test)

add_executable(testPerturbModelParameterLWR testPerturbModelParameterLWR.cpp)
target_link_libraries(testPerturbModelParameterLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testPerturbModelParameterLWR  DESTINATION bin_test)

add_executable(testBasisFunctionsLWR testBasisFunctionsLWR.cpp)
target_link_libraries(testBasisFunctionsLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testBasisFunctionsLWR  DESTINATION bin_test)
