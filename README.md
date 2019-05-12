# What?

This repository provides an implementation of dynamical systems, 
function approximators, 
[dynamical movement primitives](http://www-clmc.usc.edu/Resources/Details?id=2663), and black-box optimization
with evolution strategies, in particular the optimization of the parameters
of dynamical movement primitives.

A PDF tutorial on these topics (generated with Doxygen) is available at https://github.com/stulp/dmpbbo/blob/master/docs/tutorial.pdf

A snapshot of the complete HTML doxygen documentation is here: http://freekstulp.net/dmpbbo/html/

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

# How?

How to install the libraries/binaries/documentation is described in [INSTALL.md](INSTALL.md)

To learn how to use the code, the first thing to do is look at the
documentation and tutorial here:

+ `build_dir/docs/html/index.html` This documentation must first be generated with doxygen, see INSTALL.txt 

+ `docs/tutorial.pdf` This is a snapshot of the PDF in docs/tutorial/

To delve into the code a bit deeper, each module has a set of demos, e.g.

+ `demos/dynamicalsystems/`
  The demos do not show all the functionality, but are well
  documented and a good place to understand how the code can be 
  used. There are python scripts that call the right executables, and
  do some plotting.

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


# Why Python and C++?

A part of the functionality of the C++ code has been mirrored in Python. The Python version is probably the better language for getting to know dmpbbo (especially if you do not know C++ ;-)  The C++ code is the better choice if you want to run dmpbbo on a real robot in a real-time environment.

# Publication

If you use this library in the context of experiments for a scientific paper, we would appreciate if you could cite this library in the paper as follows:

    @MISC{stulp_dmpbbo,
	author = {Freek Stulp, Gennaro},
	title  = {DmpBbo: A versatile Python/C++ library for Function Approximation, Dynamical Movement Primitives, and Black-Box Optimization},
	year   = {2019},
	doi    = {10.21105/joss.01225},
	url    = {https://www.theoj.org/joss-papers/joss.01225/10.21105.joss.01225.pdf}
    }

Link to the paper [PDF](https://www.theoj.org/joss-papers/joss.01225/10.21105.joss.01225.pdf).

### Build Status

[![Build Status](https://travis-ci.org/stulp/dmpbbo.svg?branch=master)](https://travis-ci.org/stulp/dmpbbo)


