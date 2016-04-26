add_executable(demoEvolutionaryOptimization demoEvolutionaryOptimization.cpp)
target_link_libraries(demoEvolutionaryOptimization bbo ${Boost_LIBRARIES})
install(TARGETS demoEvolutionaryOptimization DESTINATION bin)



