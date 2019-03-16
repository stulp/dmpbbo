\ref DynamicalSystems

\li demoExponentialSystem.cpp:  Demonstrates how to initialize and integrate an exponential dynamical system.
\li demoDynamicalSystems.cpp:  Demonstrates how to initialize, integrate, perturb all implemented exponential systems.

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

