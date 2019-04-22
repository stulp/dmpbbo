# What?

This repository provides an implementation of dynamical systems, 
function approximators, 
[dynamical movement primitives](http://www-clmc.usc.edu/Resources/Details?id=2663), and black-box optimization
with evolution strategies, in particular the optimization of the parameters
of dynamical movement primitives.


# For whom?

This library may be useful for you if you

+ are interested in the theory behind dynamical movement primitives and their optimization. Then the <a href="tutorial/"><b>tutorials</b></a> are the best place to start.

+ already know about dynamical movement primitives and reinforcement learning, but would rather use existing, tested code than brew it yourself. In this case, <a href="demos_cpp/"><b>demos_cpp/</b></a> and <a href="demos_python/"><b>demos_python/</b></a> are a good starting point, as they provide examples of how to use the code.

+ run the optimization of DMPs on a real robot. In this case, go right ahead to <a href="demo_robot/"><b>demo_robot/</b></a>.

+ want to contribute. If you want to delve deeper into the functionality of the code, the **doxygen documentation of the API** is for you. See the [INSTALL.md](INSTALL.md) on how to generate it.

  
 
# How?

How to install the libraries/binaries/documentation is described in [INSTALL.md](INSTALL.md)


# Code structure

Most submodules of this project are independent of all others, so if you don't care 
about dynamical movement primitives, the following submodules can still easily be 
integrated in other code to perform some (hopefully) useful function:

+ `functionapproximators/` : a module that defines a generic interface for function 
  approximators, as well as several specific implementations (LWR, LWPR, iRFRLS, GMR)
    
+ `dynamicalsystems/` : a module that defines a generic interface for dynamical 
  systems, as well as several specific implementations (exponential, sigmoid, 
  spring-damper)

+ `bbo/` : implementation of some (rather simple) algorithms for the stochastic 
  optimization of black-box cost functions

  
## Why Python and C++?

When optimizing DMPs on a real robot, it's best to have the DMPs running in your real-time control loop. Hence, DMPs need to be implemented in C++. For the optimization algorithms itself, real-time concerns are not an issue. However, on-the-fly visualization to monitor the optimization process is important, and for this Python is a better choice.

For completeness, basic DMP functionality has been implemented in Python as well. And the optimization algorithms have been implemented in C++ also. However, the main use case is C++ for DMPs, and Python for optimization. How to do this is implemented in `demo_robot/`, and documented in `tutorial/dmp_bbo_robot.md`

Note that for now the Python code has not been documented well, please Doxygen navigate the C++ documentation instead (class/function names have been kept consistent).

# Why dmpbbo?

For our own use, the aims of coding this were the following:

+ Allowing easy and modular exchange of different dynamical systems within 
  dynamical movement primitives.

+ Allowing easy and modular exchange of different function approximators within 
  dynamical movement primitives.
    
+ Being able to compare different exploration strategies (e.g. covariance matrix 
  adaptation vs. exploration decay) when optimizing dynamical movement primitives.
    
+ Enabling the optimization of different parameter subsets of function approximators.
    
+ Running dynamical movement primitives on real robots.

##  Research background

In 2014, I decided to write one library that integrates the different research threads on the acquisition and optimization that I had been pursuing since 2009. These threads are listed below. Also, I wanted to provide a tutorial on dynamical movement primitives for students, along with code to try DMPs out in practice.

* Representation and training of parameterized skills, i.e. motion primitives that adapt their trajectory to task parameters [@matsubara11learning], [@silva12learning],  [@stulp13learning].

* Representing and optimizing gain schedules and force profiles as part of a DMP [@buchli11learning], [@kalakrishnan11learning]


*  Showing that evolution strategies outperform reinforcement learning algorithms when optimizing the parameters of a DMP [@stulp13robot], [@stulp12policy_hal]

* Demonstrating the advantages of using covariance matrix adaptation for the policy improvement [@stulp12path],[@stulp12adaptive],[@stulp14simultaneous]

* Using the same unified model for the model parameters of different function approximators [@stulp15many]. In fact, coding this library lead to this article, rather than vice versa.

If you use this library in the context of experiments for a scientific paper, we would appreciate if you could cite this library in the paper as follows:

    @MISC{stulp_dmpbbo,
        author = {Freek Stulp},
        title  = {{\tt DmpBbo} -- A C++ library for black-box optimization of 
                                                    dynamical movement primitives.},
        year   = {2014},
        url    = {https://github.com/stulp/dmpbbo.git}
    }


# Contributing

Contributions in the form of feedback, code, and bug reports are very welcome:

* If you have found an issue or a bug, please open a GitHub issue.
* If you want to implement a new feature, please fork the source code, modify, and issue a pull request through the project GitHub page.

# Build Status

[![Build Status](https://travis-ci.org/stulp/dmpbbo.svg?branch=master)](https://travis-ci.org/stulp/dmpbbo)




