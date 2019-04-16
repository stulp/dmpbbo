#ifndef _SERIALIZATION_BBO_H_
#define _SERIALIZATION_BBO_H_

#include <boost/serialization/export.hpp>

#include "bbo/DistributionGaussian.hpp"
#include "bbo/updaters/UpdaterMean.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"

BOOST_CLASS_EXPORT(DmpBbo::DistributionGaussian);
BOOST_CLASS_EXPORT(DmpBbo::UpdaterMean);
BOOST_CLASS_EXPORT(DmpBbo::UpdaterCovarDecay);
BOOST_CLASS_EXPORT(DmpBbo::UpdaterCovarAdaptation);
                           
                           
#endif // _SERIALIZATION_BBO_H_

