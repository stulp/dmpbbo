#ifndef ENTERING_REAL_TIME_CRITICAL_CODE

#ifdef REALTIME_CHECKS

// If REALTIME_CHECKS is defined, we want to check for dynamic memory allocation.
// Make Eigen check for dynamic memory allocation
#define EIGEN_RUNTIME_NO_MALLOC
// We define ENTERING_REAL_TIME_CRITICAL_CODE and EXITING_REAL_TIME_CRITICAL_CODE to start/stop
// checking dynamic memory allocation
#define ENTERING_REAL_TIME_CRITICAL_CODE \
  bool is_malloc_allowed_before=Eigen::internal::is_malloc_allowed(); \
  Eigen::internal::set_is_malloc_allowed(false);
  
#define EXITING_REAL_TIME_CRITICAL_CODE \
  Eigen::internal::set_is_malloc_allowed(is_malloc_allowed_before);

#else // REALTIME_CHECKS

// REALTIME_CHECKS is not defined, not need to do any checks on real-time code. Simply set
// ENTERING_REAL_TIME_CRITICAL_CODE and EXITING_REAL_TIME_CRITICAL_CODE to empty strings.
#define ENTERING_REAL_TIME_CRITICAL_CODE
#define EXITING_REAL_TIME_CRITICAL_CODE

#endif // REALTIME_CHECKS

#endif // ENTERING_REAL_TIME_CRITICAL_CODE

 