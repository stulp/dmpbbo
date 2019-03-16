add_executable(demoOptimizationTask demoOptimizationTask.cpp)
target_link_libraries(demoOptimizationTask dmp_bbo bbo dmp functionapproximators dynamicalsystems ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationTask DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoOptimizationDmp demoOptimizationDmp.cpp)
target_link_libraries( demoOptimizationDmp taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationDmp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
 
add_executable(demoOptimizationDmpArm2D demoOptimizationDmpArm2D.cpp)
target_link_libraries( demoOptimizationDmpArm2D taskviapointarm2d taskviapoint tasksolverdmparm2d dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationDmpArm2D DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
 
add_executable( demoOptimizationDmpParallel demoOptimizationDmpParallel.cpp ) 
target_link_libraries( demoOptimizationDmpParallel taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES} )
install(TARGETS demoOptimizationDmpParallel DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
 
add_executable( demoImitationAndOptimization demoImitationAndOptimization.cpp )
target_link_libraries( demoImitationAndOptimization dmp functionapproximators dynamicalsystems dmp_bbo taskviapoint bbo ${LWPR_LIBRARIES} ${Boost_LIBRARIES} ) 
install(TARGETS demoImitationAndOptimization DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})



