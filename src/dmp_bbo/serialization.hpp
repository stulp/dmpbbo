#ifndef _SERIALIZATION_DPM_BBO_H_
#define _SERIALIZATION_DPM_BBO_H_

//#ifndef BOOST_ARCHIVE_XML_IARCHIVE_HPP
//#error "serialization.hpp should be included after including boost/archive/* headers!"
//#endif

//#ifndef BOOST_ARCHIVE_XML_OARCHIVE_HPP
//#error "serialization.hpp should be included after including boost/archive/* headers!"
//#endif

#include "dmp/serialization.hpp"
#include "bbo/serialization.hpp"

#include <boost/serialization/export.hpp>

#include "Rollout.hpp"
#include "Task.hpp"
#include "TaskWithTrajectoryDemonstrator.hpp"
#include "tasks/TaskSolverDmpArm2D.hpp"
#include "tasks/TaskViapointArm2D.hpp"
#include "tasks/TaskViapoint.hpp"
#include "TaskSolver.hpp"
#include "TaskSolverDmp.hpp"
#include "ExperimentBBO.hpp"

BOOST_CLASS_EXPORT(DmpBbo::Rollout);
BOOST_CLASS_EXPORT(DmpBbo::Task);
BOOST_CLASS_EXPORT(DmpBbo::TaskWithTrajectoryDemonstrator);
BOOST_CLASS_EXPORT(DmpBbo::TaskSolverDmpArm2D);
BOOST_CLASS_EXPORT(DmpBbo::TaskViapointArm2D);
BOOST_CLASS_EXPORT(DmpBbo::TaskViapoint);
BOOST_CLASS_EXPORT(DmpBbo::TaskSolver);
BOOST_CLASS_EXPORT(DmpBbo::TaskSolverDmp);
BOOST_CLASS_EXPORT(DmpBbo::ExperimentBBO);
                           
                           
#endif // _SERIALIZATION_DPM_BBO_H_












