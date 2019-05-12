# Demos

This directory contains demos to show the functionality of DmpBbo. The cpp files are intented as a basis for your own tests/demos/projects.

Many of the compiled executables are accompanied by a Python wrapper, which calls the executable, and reads the files it writes, and then plots them (yes, I know about Python bindings; this approach allows better debugging of the format of the output files, which should always remain compatible between the C++ and Python versions of DmpBbo). The easiest way to run the demos is always to call the Python wrapper, rather than the executable.

If you are not interested in C++, the pure Python demos are located in dmpbbo/demos_python/.

The bottom-up approach to understanding the functionality of the code would imply the following order (each module/directory contains its own README.md file):

* Dynamical Systems (in dynamicalsystems/) This module provides implementations of several basic dynamical systems. DMPs are combinations of such systems. This module is completely independent of all other modules.

* Function Approximation (in functionapproximators/) This module provides implementations (but mostly wrappers around external libraries) of several function approximators. DMPs use function approximators to learn and reproduce arbitrary smooth movements. This module is completely independent of all other modules.

* Dynamical Movement Primitives (in dmp/) This module provides an implementation of several types of DMPs. It depends on both the DynamicalSystems and FunctionApproximators modules, but no other.

* Black Box Optimization (in bbo/) This module provides implementations of several stochastic optimization algorithms for the optimization of black-box cost functions. This module is completely independent of all other modules.

* Black Box Optimization of Dynamical Movement Primitives (in dmp_bbo/) This module applies black-box optimization to the parameters of a DMP. It depends on all the other modules.









