#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "pydmpbbo.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(_dmpbbo)
{
    class_<UpdaterCovarAdaptation>("UpdaterCovarAdaptation", init<double, std::string, const boost::python::list&, bool, double, const boost::python::list, const boost::python::list>())
        .def("update_distribution", &UpdaterCovarAdaptation::updateDistribution)
        .def("get_mean", &UpdaterCovarAdaptation::getMean)
        .def("get_covariance", &UpdaterCovarAdaptation::getCovariance)
    ;
    class_<Dmp>("Dmp", init<int, int>())
        .def("trajectory", &Dmp::trajectory)
        .def("train", &Dmp::train)
        .def("set_tau", &Dmp::setTau)
        .def("set_initial_state", &Dmp::set_initial_state)
        .def("set_attractor_state", &Dmp::set_attractor_state)
    ;
}
