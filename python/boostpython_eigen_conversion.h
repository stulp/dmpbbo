#ifndef BOOSTPYTHON_EIGEN_CONVERSION_H
#define BOOSTPYTHON_EIGEN_CONVERSION_H

#include <boost/python.hpp>
#include <eigen3/Eigen/Core>

using namespace Eigen;



VectorXd listToVectorXd(const boost::python::list& x);

MatrixXd listToMatrixXd(const boost::python::list& x);

boost::python::list vectorXdToList(const VectorXd x);

boost::python::list matrixXdToList(const MatrixXd x);

#endif
