#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "pydmpbbo.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(_dmpbbo)
{
    class_<PyDmpBbo>("DmpBbo")
        .def("run", &PyDmpBbo::run)
    ;
}
