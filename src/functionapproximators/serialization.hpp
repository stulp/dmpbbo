#ifndef _SERIALIZATION_FUNCTION_APPROXIMATORS_H_
#define _SERIALIZATION_FUNCTION_APPROXIMATORS_H_

#include <boost/serialization/export.hpp>

#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/MetaParametersGPR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/MetaParametersRRRFF.hpp"

BOOST_CLASS_EXPORT(DmpBbo::MetaParametersGMR);
BOOST_CLASS_EXPORT(DmpBbo::MetaParametersGPR);
BOOST_CLASS_EXPORT(DmpBbo::MetaParametersLWR);
BOOST_CLASS_EXPORT(DmpBbo::MetaParametersRBFN);
BOOST_CLASS_EXPORT(DmpBbo::MetaParametersRRRFF);




#include "functionapproximators/ModelParametersGMR.hpp"
#include "functionapproximators/ModelParametersGPR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"
#include "functionapproximators/ModelParametersRRRFF.hpp"

BOOST_CLASS_EXPORT(DmpBbo::ModelParametersGMR);
BOOST_CLASS_EXPORT(DmpBbo::ModelParametersGPR);
BOOST_CLASS_EXPORT(DmpBbo::ModelParametersLWR);
BOOST_CLASS_EXPORT(DmpBbo::ModelParametersRBFN);
BOOST_CLASS_EXPORT(DmpBbo::ModelParametersRRRFF);



#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/FunctionApproximatorGPR.hpp"
#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/FunctionApproximatorRRRFF.hpp"

BOOST_CLASS_EXPORT(DmpBbo::FunctionApproximatorGMR);
BOOST_CLASS_EXPORT(DmpBbo::FunctionApproximatorGPR);
BOOST_CLASS_EXPORT(DmpBbo::FunctionApproximatorLWR);
BOOST_CLASS_EXPORT(DmpBbo::FunctionApproximatorRBFN);
BOOST_CLASS_EXPORT(DmpBbo::FunctionApproximatorRRRFF);

#ifdef USE_LWPR

#include "functionapproximators/MetaParametersLWPR.hpp"
BOOST_CLASS_EXPORT(DmpBbo::MetaParametersLWPR);
#include "functionapproximators/ModelParametersLWPR.hpp"
BOOST_CLASS_EXPORT(DmpBbo::ModelParametersLWPR);                         
#include "functionapproximators/FunctionApproximatorLWPR.hpp"
BOOST_CLASS_EXPORT(DmpBbo::FunctionApproximatorLWPR);

#endif // USE_LWPR


#endif // _SERIALIZATION_FUNCTION_APPROXIMATORS_H_