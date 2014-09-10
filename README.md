# What?

This repository provides an implementation of dynamical systems, 
function approximators, 
[dynamical movement primitives](http://www-clmc.usc.edu/Resources/Details?id=2663), and black-box optimization
with evolution strategies, in particular the optimization of the parameters
of dynamical movement primitives.

A PDF tutorial on these topics (generated with Doxygen) is available at https://github.com/stulp/dmpbbo/blob/master/docs/tutorial.pdf

The complete HTML doxygen documentation is at: http://perso.ensta-paristech.fr/~stulp/dmpbbo/

# For whom?

This library may be useful for you if you

+ are new to dynamical movement primitives and want to learn about them (see the 
    tutorial in the doxygen documentation).

+ already know about dynamical movement primitives, but would rather use existing,
    tested code than brew it yourself.
  
+ want to do reinforcement learning/optimization of dynamical movement primitives.
  
  
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
  
If you use this library in the context of experiments for a scientific paper, we would appreciate if you could cite this library in the paper as follows:

    @MISC{stulp_dmpbbo,
        author = {Freek Stulp},
        title  = {{\tt DmpBbo} -- A C++ library for black-box optimization of 
                                                    dynamical movement primitives.},
        year   = {2014},
        url    = {https://github.com/stulp/dmpbbo.git}
    }

# How?

How to install the libraries/binaries/documentation is described in INSTALL.txt

To learn how to use the code, the first thing to do is look at the
documentation and tutorial here:

+ `build_dir/docs/html/index.html` This documentation must first be generated with doxygen, see INSTALL.txt 

+ `docs/tutorial.pdf` This is a snapshot of the PDF in docs/tutorial/

To delve into the code a bit deeper, each module has a set of demos, e.g.

+ `src/dynamicalsystems/demos/`
  The demos do not show all the functionality, but are well
  documented and a good place to understand how the code can be 
  used. There are python scripts that call the right executables, and
  do some plotting.

For more advanced stuff, you can also have a look at the tests, e.g. 

+ `src/dynamicalsystems/tests/`
  These are not unit tests per se, but more debugging tools that visualize the 
  results of an experiment or parameter setting. The tests 
  are not well documented, but exploit more of the functionality of 
  the code. Note that the test binaries are only built in debug 
  mode (in bin_test)

# Why?

For our own use, the aims of coding this were the following:

+ Allowing easy and modular exchange of different dynamical systems within 
  dynamical movement primitives.

+ Allowing easy and modular exchange of different function approximators within 
  dynamical movement primitives.
    
+ Being able to compare different exploration strategies (e.g. covariance matrix 
  adaptation vs. exploration decay) when optimizing dynamical movement primitives.
    
+ Enabling the optimization of different parameter subsets of function approximators.
    
+ Running dynamical movement primitives on real robots.

### Build Status

[![Build Status](https://travis-ci.org/stulp/dmpbbo.svg?branch=master)](https://travis-ci.org/stulp/dmpbbo)


