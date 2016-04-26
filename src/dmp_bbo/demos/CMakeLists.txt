add_executable(demoEvolutionaryOptimizationTask demoEvolutionaryOptimizationTask.cpp)
target_link_libraries(demoEvolutionaryOptimizationTask dmp_bbo bbo ${Boost_LIBRARIES})
install(TARGETS demoEvolutionaryOptimizationTask DESTINATION bin)

add_executable(demoDmpBbo demoDmpBbo.cpp)
target_link_libraries( demoDmpBbo taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoDmpBbo DESTINATION bin)
 
add_executable( demoDmpBboMultiDim demoDmpBboMultiDim.cpp ) 
target_link_libraries( demoDmpBboMultiDim taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES} )
install(TARGETS demoDmpBboMultiDim DESTINATION bin)
 
add_executable( demoImitationAndOptimization demoImitationAndOptimization.cpp )
target_link_libraries( demoImitationAndOptimization dmp functionapproximators dynamicalsystems dmp_bbo taskviapoint bbo ${LWPR_LIBRARIES} ${Boost_LIBRARIES} ) 
install(TARGETS demoImitationAndOptimization DESTINATION bin)



