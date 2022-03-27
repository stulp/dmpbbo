add_executable(testJson testJson.cpp)
target_link_libraries(testJson functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testTrainingCompareCppPython testTrainingCompareCppPython.cpp)
target_link_libraries(testTrainingCompareCppPython functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorCloning testFunctionApproximatorCloning.cpp)
target_link_libraries(testFunctionApproximatorCloning functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorRealtime testFunctionApproximatorRealtime.cpp)
target_link_libraries(testFunctionApproximatorRealtime functionapproximators ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorSerialization testFunctionApproximatorSerialization.cpp)
target_link_libraries(testFunctionApproximatorSerialization functionapproximators ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(testModelParameterValues testModelParameterValues.cpp)
target_link_libraries(testModelParameterValues functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testToUnifiedModel testToUnifiedModel.cpp)
target_link_libraries(testToUnifiedModel functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})



################################################################################
# LWR
add_executable(testMetaParametersLWR testMetaParametersLWR.cpp)
target_link_libraries(testMetaParametersLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testPerturbModelParameterLWR testPerturbModelParameterLWR.cpp)
target_link_libraries(testPerturbModelParameterLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testBasisFunctionsLWR testBasisFunctionsLWR.cpp)
target_link_libraries(testBasisFunctionsLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})


################################################################################
# GMR
add_executable(testSaveGMMMatrix testSaveGMMMatrix.cpp)
target_link_libraries(testSaveGMMMatrix functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testMultiDimOutput testMultiDimOutput.cpp)
target_link_libraries(testMultiDimOutput functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testIncrementalTrainingGMR testIncrementalTrainingGMR.cpp)
target_link_libraries(testIncrementalTrainingGMR functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

