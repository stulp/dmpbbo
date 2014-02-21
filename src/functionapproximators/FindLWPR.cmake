find_path(LWPR_INCLUDE_DIR lwpr.hh HINTS /usr/local/include/)
find_library(LWPR_LIBRARY NAMES lwpr HINTS /usr/local/lib/)

find_package(PkgConfig)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LWPR REQUIRED_VARS LWPR_LIBRARY LWPR_INCLUDE_DIR)

if (LWPR_FOUND)
  set(LWPR_LIBRARIES ${LWPR_LIBRARY} )
  set(LWPR_INCLUDE_DIRS ${LWPR_INCLUDE_DIR} )
endif()

#message("      LWPR_INCLUDE_DIR  = ${LWPR_INCLUDE_DIR}")
#message("      LWPR_INCLUDE_DIRS = ${LWPR_INCLUDE_DIRS}")
#message("      LWPR_LIBRARY      = ${LWPR_LIBRARY}")
#message("      LWPR_LIBRARIES    = ${LWPR_LIBRARIES}")
#message("      LWPR_FOUND        = ${LWPR_FOUND}")


