# DynamicalSystems Demos

## demoExponentialSystem.cpp / demoExponentialSystemWrapper.py  

Demonstrates how to initialize and integrate an exponential dynamical system. One (optional) argument is interpreted to be a directory to which to store the result. 

The Python wrapper has the same interface, but uses a default directory.

## demoDynamicalSystems.cpp / demoDynamicalSystems.py  

Demonstrates how to initialize, integrate, perturb all implemented exponential systems.

The executable demoDynamicalSystems is called with a directory, and one or several of the following flags for integration: 

- "rungekutta" - Use 4th-order Runge-Kutta integration (more accurate, but more calls of DynamicalSystem::differentialEquation)
- "euler"      - Use simple Euler integration (less accurate, but faster)
- "analytical" - Use the analytical solution instead of numerical integration
- "tau"        - Change tau before doing numerical integration
- "attractor"  - Change the attractor state during numerical integration
- "perturb"    - Perturb the state during numerical integration

The results are stored in the directory that is passed as the first argument.

The Python wrapper has the same interface, but uses a default directory.
