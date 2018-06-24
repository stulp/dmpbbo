file(GLOB SOURCES_TASK TaskViapoint.cpp) 
add_library(taskviapoint ${SHARED_OR_STATIC} ${SOURCES_TASK})
install(TARGETS taskviapoint DESTINATION ${LIB_INSTALL_DIR})
file(GLOB HEADERS_TASK TaskViapoint.hpp)
install(FILES ${HEADERS_TASK} DESTINATION ${INCLUDE_INSTALL_DIR}/dmp_bbo/tasks)

file(GLOB SOURCES_TASK TaskViapointArm2D.cpp) 
add_library(taskviapointarm2d ${SHARED_OR_STATIC} ${SOURCES_TASK})
install(TARGETS taskviapointarm2d DESTINATION ${LIB_INSTALL_DIR})
file(GLOB HEADERS_TASK TaskViapointArm2D.hpp)
install(FILES ${HEADERS_TASK} DESTINATION ${INCLUDE_INSTALL_DIR}/dmp_bbo/tasks)

file(GLOB SOURCES_TASK TaskSolverDmpArm2D.cpp) 
add_library(tasksolverdmparm2d ${SHARED_OR_STATIC} ${SOURCES_TASK})
install(TARGETS tasksolverdmparm2d DESTINATION ${LIB_INSTALL_DIR})
file(GLOB HEADERS_TASK TaskSolverDmpArm2D.hpp)
install(FILES ${HEADERS_TASK} DESTINATION ${INCLUDE_INSTALL_DIR}/dmp_bbo/tasks)

