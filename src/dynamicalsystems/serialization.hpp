#ifndef _SERIALIZATION_DYNAMICAL_SYSTEMS_H_
#define _SERIALIZATION_DYNAMICAL_SYSTEMS_H_

#include <boost/serialization/export.hpp>

#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

BOOST_CLASS_EXPORT(DmpBbo::ExponentialSystem);
BOOST_CLASS_EXPORT(DmpBbo::SigmoidSystem);
BOOST_CLASS_EXPORT(DmpBbo::TimeSystem);
BOOST_CLASS_EXPORT(DmpBbo::SpringDamperSystem);


#endif // _SERIALIZATION_DYNAMICAL_SYSTEMS_H_