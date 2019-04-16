#ifndef _SERIALIZATION_BBO_H_
#define _SERIALIZATION_BBO_H_

#include <boost/serialization/export.hpp>

#include "dmp/DistributionGaussian.hpp"
#include "dmp/UpdaterMean.hpp"
#include "dmp/UpdaterCovarDecay.hpp"
#include "dmp/UpdaterCovarAdaptation.hpp"

BOOST_CLASS_EXPORT(DmpBbo::DistributionGaussian);
BOOST_CLASS_EXPORT(DmpBbo::UpdaterMean);
BOOST_CLASS_EXPORT(DmpBbo::UpdaterCovarDecay);
BOOST_CLASS_EXPORT(DmpBbo::UpdaterCovarAdaptation);
                           
                           
#endif // _SERIALIZATION_BBO_H_

