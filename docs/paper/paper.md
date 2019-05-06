---
title: 'DmpBbo: A versatile Python/C++ library for Function Approximation, Dynamical Movement Primitives, and Black-Box Optimization'
tags:
  - Robotics
  - Motion Primitives
  - Machine Learning
authors:
 - name: Freek Stulp
   orcid: 0000-0001-9555-9517
   affiliation: 1
 - name: Gennaro Raiola
   orcid: 0000-0003-1481-1106
   affiliation: 1, 2
affiliations:
 - name: ENSTA-ParisTech (at the time of this work)
   index: 1
 - name: Department of Advanced Robotics, Istituto Italiano di Tecnologia (IIT)
   index: 2
date: 16 January 2018
bibliography: paper.bib
---
# General overview

Dynamical movement primitives (DMPs) [@ijspeert02movement,@ijspeert13dynamical] are one of the most popular representations for goal-directed motion primitives in robotics. They are also often used as the policy representation for policy improvement in robotics, a particular form of reinforcement learning. `dmpbbo` provides five software modules for the representation and optimization of dynamical movement primitives. These five modules are:

* `dynamicalsystems/`,  various dynamical systems representing for instance exponential decay or spring-damper systems (standalone module).
* `functionapproximators/`, various function approximators such as Gaussian process regression, radial basis function networks, and Gaussian mixture regression (standalone module).
* `dmp/`, implementation of dynamical movement primitives, where various dynamical systems and function approximators in the first modules can be easily exchanged to get DMPs with different properties.
* `bbo/`, implementations of several stochastic optimization algorithms for the optimization of black-box cost functions (standalone module)
* `dmp_bbo/`, applies black-box optimization to the parameters of a DMP (depends on all other modules)

`dmpbbo` provides both a real-time C++ implementation, as well as an implementation in Python for non-roboticists.

`dmpbbo` is accompanied by an extensive tutorial on the motivation for dynamical movement primitives, and their mathematical derivation. 

## Advanced features

Several more advanced features implemented in dmpbbo are:

* Contextual dynamical movement primitives, which can adapt to variations of tasks [@stulp13learning]

* Dynamical movement primitives with gain schedules [@buchli11learning]

* Unified models for function approximators [@stulp15many]

* Covariance matrix adaptation in black-box optimization, which enables automatic exploration tuning [@stulp12adaptive]


## Applications

This library and its predecessors were used in the following scientific publications [@stulp12adaptive,@stulp13learning,stulp14simultaneous,@stulp15many]. The images below are snapshots of robotic applications where `dmpbbo' was used. And here a list of videos:

* https://www.youtube.com/watch?v=R7LWkh1UMII
* https://www.youtube.com/watch?v=MAiw3Ke7bh8
* https://www.youtube.com/watch?v=jkaRO8J_1XI
* https://www.youtube.com/watch?v=i_JBRojCqcc

![Overview](images/robots.png)

Robot names and credits in order of appearance: iCub (Photo by ISIR), MEKA (Photo by ENSTA ParisTech), Pepper (Photo by SoftBank)

# References
