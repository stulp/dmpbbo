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


================================================================
Dynamical systems

\subsection dynsys_implementation1_plotting Plotting

If you save the output of a dynamical in a file with format (where D is the dimensionality of the system, and T is the number of time steps)
\verbatim
t_0   x^0_0 x^1_0 .. x^D_0   xd^0_0 xd^1_0 .. xd^D_0     
t_1   x^0_1 x^1_1 .. x^D_1   xd^0_1 xd^1_1 .. xd^D_1     
 :       :     :       :         :      :       :        
t_T   x^0_T x^1_T .. x^D_T   xd^0_T xd^1_T .. xd^D_T     
\endverbatim
you can plot this output with 
\code
python dynamicalsystems/plotting/plotDynamicalSystem.py file.txt
\endcode

\subsection dynsys_implementation1_demo Demos

A demonstration of how to initialize and integrate an ExponentialSystem is in demoExponentialSystem.cpp

A more complete demonstration including all implemented dynamical systems is in demoDynamicalSystems.cpp. If you call the resulting executable with a directory argument, e.g.
\code
./demoDynamicalSystems /tmp/demoDynamicalSystems
\endcode
it will save results to file, which you can plot with for instance:
\code
python plotDynamicalSystem.py /tmp/demoDynamicalSystems/ExponentialSystem/results_rungekutta.txt
python plotDynamicalSystem.py /tmp/demoDynamicalSystems/ExponentialSystem/results_euler.txt
\endcode
Different test can be performed with the dynamical system. The test can be chosen by passing further argument, e.g. 
\code
./demoDynamicalSystems /tmp/demoDynamicalSystems rungekutta euler
\endcode
will integrate the dynamical systems with both the Runge-Kutta and simple Euler method. The available tests are:
\li "rungekutta" - Use 4th-order Runge-Kutta integration (more accurate, but more calls of DynamicalSystem::differentialEquation)
\li "euler"      - Use simple Euler integration (less accurate, but faster)
\li "analytical" - Use the analytical solution instead of numerical integration
\li "tau"        - Change tau before doing numerical integration
\li "attractor"  - Change the attractor state during numerical integration
\li "perturb"    - Perturb the state during numerical integration

To compare for instance the analytical solution with the Runge-Kutta integration in a plot, you can do
\code
python plotDynamicalSystemComparison.py /tmp/demoDynamicalSystems/ExponentialSystem/results_analytical.txt  /tmp/demoDynamicalSystems/ExponentialSystem/results_rungekutta.txt
\endcode

