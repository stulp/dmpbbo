add_executable(demoOptimizationTask demoOptimizationTask.cpp)
target_link_libraries(demoOptimizationTask dmp_bbo bbo dmp functionapproximators dynamicalsystems eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationTask DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoOptimizationDmp demoOptimizationDmp.cpp)
target_link_libraries( demoOptimizationDmp taskviapoint dmp_bbo dmp functionapproximators eigen dynamicalsystems bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationDmp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
 
add_executable(demoOptimizationDmpArm2D demoOptimizationDmpArm2D.cpp)
target_link_libraries( demoOptimizationDmpArm2D taskviapointarm2d taskviapoint tasksolverdmparm2d dmp_bbo dmp functionapproximators eigen dynamicalsystems bbo  ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS demoOptimizationDmpArm2D DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
 
add_executable( demoOptimizationDmpParallel demoOptimizationDmpParallel.cpp ) 
target_link_libraries( demoOptimizationDmpParallel taskviapoint dmp_bbo dmp functionapproximators dynamicalsystems bbo  eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES} )
install(TARGETS demoOptimizationDmpParallel DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_executable( demoImitationAndOptimization demoImitationAndOptimization.cpp )
target_link_libraries( demoImitationAndOptimization taskviapoint dmp_bbo dmp functionapproximators dynamicalsystems bbo eigen ${LWPR_LIBRARIES} ${Boost_LIBRARIES} ) 
install(TARGETS demoImitationAndOptimization DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})



