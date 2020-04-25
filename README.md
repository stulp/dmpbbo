[![DOI](http://joss.theoj.org/papers/10.21105/joss.01225/status.svg)](https://doi.org/10.21105/joss.01225)

[![Build Status](https://travis-ci.org/stulp/dmpbbo.svg?branch=master)](https://travis-ci.org/stulp/dmpbbo)

# What?

This repository provides an implementation of dynamical systems, 
function approximators, 
[dynamical movement primitives](http://www-clmc.usc.edu/Resources/Details?id=2663), and black-box optimization
with evolution strategies, in particular the optimization of the parameters
of dynamical movement primitives.


# For whom?

This library may be useful for you if you

+ are interested in the **theory** behind dynamical movement primitives and their optimization. Then the <a href="tutorial/"><b>tutorials</b></a> are the best place to start.

+ already know about dynamical movement primitives and reinforcement learning, but would rather **use existing, tested code** than brew it yourself. In this case, <a href="demos_cpp/"><b>demos_cpp/</b></a> and <a href="demos_python/"><b>demos_python/</b></a> are a good starting point, as they provide examples of how to use the code.

+ run the optimization of DMPs **on a real robot**. In this case, go right ahead to <a href="demo_robot/"><b>demo_robot/</b></a>.

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

    @article{stulp2019dmpbbo,
	author  = {Freek Stulp and Gennaro Raiola},
	title   = {DmpBbo: A versatile Python/C++ library for Function Approximation, Dynamical Movement Primitives, and Black-Box Optimization},
	journal = {Journal of Open Source Software}
	year    = {2019},
	doi     = {10.21105/joss.01225},
	url     = {https://www.theoj.org/joss-papers/joss.01225/10.21105.joss.01225.pdf}
    }

## Bibliography

* <a id="buchli11learning"></a><b>[buchli11learning]</b>  Jonas Buchli, Freek Stulp, Evangelos Theodorou, and Stefan Schaal. <a href="http://ijr.sagepub.com/content/early/2011/03/31/0278364911402527">Learning variable impedance control</a>. <em>International Journal of Robotics Research</em>, 30(7):820-833, 2011.
* <a id="ijspeert02movement"></a><b>[ijspeert02movement]</b>  A. J. Ijspeert, J. Nakanishi, and S. Schaal. Movement imitation with nonlinear dynamical systems in humanoid robots. In <em>Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)</em>, 2002.
* <a id="ijspeert13dynamical"></a><b>[ijspeert13dynamical]</b>  A. Ijspeert, J. Nakanishi, P Pastor, H. Hoffmann, and S. Schaal. Dynamical Movement Primitives: Learning attractor models for motor behaviors. <em>Neural Computation</em>, 25(2):328-373, 2013.
* <a id="kalakrishnan11learning"></a><b>[kalakrishnan11learning]</b>  M. Kalakrishnan, L. Righetti, P. Pastor, and S. Schaal. <a href="http://www-clmc.usc.edu/publications/K/kalakrishnan-IROS2011">Learning force control policies for compliant manipulation</a>. In <em>IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2011)</em>, 2011.
* <a id="kulvicius12joining"></a><b>[kulvicius12joining]</b>  Tomas Kulvicius, KeJun Ning, Minija Tamosiunaite, and Florentin Wörgötter. Joining movement sequences: Modified dynamic movement primitives for robotics applications exemplified on handwriting. <em>IEEE Transactions on Robotics</em>, 28(1):145-157, 2012.
* <a id="matsubara11learning"></a><b>[matsubara11learning]</b>  T Matsubara, S Hyon, and J Morimoto. Learning parametric dynamic movement primitives from multiple demonstrations. <em>Neural Networks</em>, 24(5):493-500, 2011.
* <a id="silva12learning"></a><b>[silva12learning]</b>  Bruno da Silva, George Konidaris, and Andrew G. Barto. Learning parameterized skills. In John Langford and Joelle Pineau, editors, <em>Proceedings of the 29th International Conference on Machine Learning (ICML-12)</em>, ICML '12, pages 1679-1686, New York, NY, USA, July 2012. Omnipress.
* <a id="stulp12adaptive"></a><b>[stulp12adaptive]</b>  Freek Stulp. Adaptive exploration for continual reinforcement learning. In <em>International Conference on Intelligent Robots and Systems (IROS)</em>, pages 1631-1636, 2012.
* <a id="stulp12path"></a><b>[stulp12path]</b>  Freek Stulp and Olivier Sigaud. Path integral policy improvement with covariance matrix adaptation. In <em>Proceedings of the 29th International Conference on Machine Learning (ICML)</em>, 2012.
* <a id="stulp12policy_hal"></a><b>[stulp12policy_hal]</b>  Freek Stulp and Olivier Sigaud. <a href="http://hal.archives-ouvertes.fr/hal-00738463">Policy improvement methods: Between black-box optimization and episodic reinforcement learning</a>. hal-00738463, 2012.
* <a id="stulp13learning"></a><b>[stulp13learning]</b>  Freek Stulp, Gennaro Raiola, Antoine Hoarau, Serena Ivaldi, and Olivier Sigaud. Learning compact parameterized skills with a single regression. In <em>IEEE-RAS International Conference on Humanoid Robots</em>, 2013.
* <a id="stulp13robot"></a><b>[stulp13robot]</b>  Freek Stulp and Olivier Sigaud. Robot skill learning: From reinforcement learning to evolution strategies. <em>Paladyn. Journal of Behavioral Robotics</em>, 4(1):49-61, September 2013.
* <a id="stulp14simultaneous"></a><b>[stulp14simultaneous]</b>  Freek Stulp, Laura Herlant, Antoine Hoarau, and Gennaro Raiola. Simultaneous on-line discovery and improvement of robotic skill options. In <em>International Conference on Intelligent Robots and Systems (IROS)</em>, 2014.
* <a id="stulp15many"></a><b>[stulp15many]</b>  Freek Stulp and Olivier Sigaud. <a href="http://www.sciencedirect.com/science/article/pii/S0893608015001185">Many regression algorithms, one unified model - a review</a>. <em>Neural Networks</em>, 2015.    

# Contributing

Contributions in the form of feedback, code, and bug reports are very welcome:

* If you have found an issue or a bug, please open a GitHub issue.
* If you want to implement a new feature, please fork the source code, modify, and issue a pull request through the project GitHub page.
