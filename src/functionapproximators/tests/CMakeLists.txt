add_executable(testJson testJson.cpp)
target_link_libraries(testJson functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testTrainingCompareCppPython testTrainingCompareCppPython.cpp)
target_link_libraries(testTrainingCompareCppPython functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorCloning testFunctionApproximatorCloning.cpp)
target_link_libraries(testFunctionApproximatorCloning functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorRealtime testFunctionApproximatorRealtime.cpp)
target_link_libraries(testFunctionApproximatorRealtime functionapproximators eigen ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(testFunctionApproximatorSerialization testFunctionApproximatorSerialization.cpp)
target_link_libraries(testFunctionApproximatorSerialization functionapproximators eigen ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(testModelParameterValues testModelParameterValues.cpp)
target_link_libraries(testModelParameterValues functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testToUnifiedModel testToUnifiedModel.cpp)
target_link_libraries(testToUnifiedModel functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})



################################################################################
# LWR
add_executable(testMetaParametersLWR testMetaParametersLWR.cpp)
target_link_libraries(testMetaParametersLWR functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testPerturbModelParameterLWR testPerturbModelParameterLWR.cpp)
target_link_libraries(testPerturbModelParameterLWR functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testBasisFunctionsLWR testBasisFunctionsLWR.cpp)
target_link_libraries(testBasisFunctionsLWR functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})


################################################################################
# GMR
add_executable(testSaveGMMMatrix testSaveGMMMatrix.cpp)
target_link_libraries(testSaveGMMMatrix functionapproximators eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testMultiDimOutput testMultiDimOutput.cpp)
target_link_libraries(testMultiDimOutput functionapproximators eigen ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

add_executable(testIncrementalTrainingGMR testIncrementalTrainingGMR.cpp)
target_link_libraries(testIncrementalTrainingGMR functionapproximators eigen ${Boost_LIBRARIES} ${LWPR_LIBRARIES})

