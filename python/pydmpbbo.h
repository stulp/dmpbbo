#ifndef PY_DMPBBO_H
#define PY_DMPBBO_H

#include <boost/python.hpp>
#include <string>

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

//#include "dynamicalsystems/DynamicalSystem.hpp"
//#include "dynamicalsystems/ExponentialSystem.hpp"
//#include "dynamicalsystems/SigmoidSystem.hpp"
//#include "dynamicalsystems/TimeSystem.hpp"
//#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <fstream>


class PyDmpBbo {
public:
    PyDmpBbo();
    void run(double tau, int n_time_steps, int n_basis_functions, int input_dim, double intersection, const std::string &save_dir);

private:
    ;
};

#endif
