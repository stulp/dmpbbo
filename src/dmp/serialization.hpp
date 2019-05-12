#ifndef _SERIALIZATION_DMPS_H_
#define _SERIALIZATION_DMPS_H_

#include "functionapproximators/serialization.hpp"
#include "dynamicalsystems/serialization.hpp"


#include <boost/serialization/export.hpp>

#include "dmp/DmpContextual.hpp"
#include "dmp/DmpContextualOneStep.hpp"
#include "dmp/DmpContextualTwoStep.hpp"
#include "dmp/Dmp.hpp"
#include "dmp/DmpWithGainSchedules.hpp"
#include "dmp/Trajectory.hpp"            

BOOST_CLASS_EXPORT(DmpBbo::DmpContextual);
BOOST_CLASS_EXPORT(DmpBbo::DmpContextualOneStep);
BOOST_CLASS_EXPORT(DmpBbo::DmpContextualTwoStep);
BOOST_CLASS_EXPORT(DmpBbo::Dmp);                    
BOOST_CLASS_EXPORT(DmpBbo::DmpWithGainSchedules);
BOOST_CLASS_EXPORT(DmpBbo::Trajectory);            
                           
                           
#endif // _SERIALIZATION_DMPS_H_

