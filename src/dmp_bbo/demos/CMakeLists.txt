add_executable(demoDmpBbo demoDmpBbo.cpp)
target_link_libraries( demoDmpBbo dmp functionapproximators dynamicalsystems dmp_bbo taskviapoint bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoDmpBbo DESTINATION bin)
 
add_executable( demoDmpBboMultiDim demoDmpBboMultiDim.cpp ) 
target_link_libraries( demoDmpBboMultiDim dmp functionapproximators dynamicalsystems dmp_bbo taskviapoint bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES} )
install(TARGETS demoDmpBboMultiDim DESTINATION bin)
 
add_executable( demoImitationAndOptimization demoImitationAndOptimization.cpp )
target_link_libraries( demoImitationAndOptimization dmp functionapproximators dynamicalsystems dmp_bbo taskviapoint bbo ${LWPR_LIBRARIES} ${Boost_LIBRARIES} ) 
install(TARGETS demoImitationAndOptimization DESTINATION bin)



