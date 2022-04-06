add_executable(demoExponentialSystem demoExponentialSystem.cpp)
target_link_libraries(demoExponentialSystem dynamicalsystems eigen ${Boost_LIBRARIES})
install(TARGETS demoExponentialSystem DESTINATION  ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demoDynamicalSystems demoDynamicalSystems.cpp)
target_link_libraries(demoDynamicalSystems dynamicalsystems eigen ${Boost_LIBRARIES})
install(TARGETS demoDynamicalSystems DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})