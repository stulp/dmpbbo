add_executable(demoOptimizationTask demoOptimizationTask.cpp)
target_link_libraries(demoOptimizationTask dmp_bbo bbo ${Boost_LIBRARIES})
install(TARGETS demoOptimizationTask DESTINATION bin)

add_executable(demoOptimizationDmp demoOptimizationDmp.cpp)
target_link_libraries( demoOptimizationDmp taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationDmp DESTINATION bin)
 
add_executable( demoOptimizationDmpMultiDim demoOptimizationDmpMultiDim.cpp ) 
target_link_libraries( demoOptimizationDmpMultiDim taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES} )
install(TARGETS demoOptimizationDmpMultiDim DESTINATION bin)
 
add_executable( demoImitationAndOptimization demoImitationAndOptimization.cpp )
target_link_libraries( demoImitationAndOptimization dmp functionapproximators dynamicalsystems dmp_bbo taskviapoint bbo ${LWPR_LIBRARIES} ${Boost_LIBRARIES} ) 
install(TARGETS demoImitationAndOptimization DESTINATION bin)



