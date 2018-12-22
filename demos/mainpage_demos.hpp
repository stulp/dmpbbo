/**
 * @file mainpage_demos.hpp
 * @brief File containing only documentation for the demos (for the Doxygen mainpage)
 * @author Freek Stulp
 */

/** \page page_demos Demos

This page provides a list of all the demos, in the recommended order. 

\section sec_demos_python_cpp Python/C++

The demos for the C++ code are in the demos/ directory, and for Python in the demos_python/ directory. For most of the C++ demos (e.g. demoExponentialSystem.cpp) there is a corresponding Python file (e.g. demoExponentialSystemWrapper.py), which essentially calls a binary (e.g. bin/demoExponentialSystem), and plots the results. (yes, I know about Python bindings; historical reasons)

Note that there are also test (e.g. ./src/dmp/dynamicalsystems/testDynamicalSystemSerialization.cpp), which are intended for debugging purposes, not to understand the code. These test are not well-documented, not included in the Doxygen documentation.

\section sec_demos_list  List of demos

Here is a list of the available demos, in the recommended order 

\ref FunctionApproximators

\li demoLeastSquares.cpp: Demonstrates how to run least squares regression
\li demoFunctionApproximatorTraining.cpp: Demonstrates how to train all functionapproximators with 1D and 2D data.

\ref DynamicalSystems

\li demoExponentialSystem.cpp:  Demonstrates how to initialize and integrate an exponential dynamical system.
\li demoDynamicalSystems.cpp:  Demonstrates how to initialize, integrate, perturb all implemented exponential systems.

\ref Dmps

\li demoDmp.cpp:  Demonstrates how to initialize, train and integrate a Dmp.
\li demoDmpTrainFromTrajectoryFile.cpp:  Demonstrates how to train a Dmp with a trajectory in a txt file.
\li demoDmpChangeGoal.cpp:  Demonstrates how to change the goal for a Dmp, and the effects of different scaling approaches.

\li demoDmpContextual.cpp:  Demonstrates how to initialize, train and integrate a Contextual Dmp.
\li demoDmpContextualGoal.cpp:  Demonstrates how to initialize, train and integrate a Contextual Dmp that adapts the goal state to the task parameters.


\ref BBO

\li demoOptimization.cpp:  Demonstrates how to run an evolution strategy to optimize a distance function, implemented as a CostFunction.


\ref DMP_BBO

\li demoOptimizationTask.cpp:  Demonstrates how to run an evolution strategy to optimize the parameters of a quadratic function, implemented as a Task and TaskSolver.
\li demoOptimizationDmp.cpp:  Demonstrates how to run an evolution strategy to optimize a Dmp.
\li demoImitationAndOptimization.cpp:  Demonstrates how to initialize a DMP with a trajectory, and then optimize it with an evolution strategy.
\li demoOptimizationDmpArm2D.cpp:  Demonstrates how to run an evolution strategy to optimize a Dmp, on a task with a viapoint task with a N-DOF arm in a 2D space.

 */
