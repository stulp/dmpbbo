/**
 * \file eigen_realtime_check.hpp
 * \author Freek Stulp
 * \brief  Header file for adding real-time debugging using several macros 
 *
 * This file is part of DmpBbo, a set of libraries and programs for the 
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 * 
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 * 
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

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

 