/**
 * @file mainpage.hpp
 * @brief File containing only documentation (for the Doxygen mainpage)
 * @author Freek Stulp
 */

/** \mainpage 

This library contains several modules for training dynamical movement primitives (DMPs), and optimizing their parameters through black-box optimization. Each module has its own dedicated page.

\li  \ref page_dyn_sys (in dynamicalsystems/) This module provides implementations of several basic dynamical systems. DMPs are combinations of such systems. This module is completely independent of all other modules.

\li \ref page_func_approx (in functionapproximators/) This module provides implementations (but mostly wrappers around external libraries) of several function approximators. DMPs use function approximators to learn and reproduce arbitrary smooth movements. This module is completely independent of all other modules.

\li \ref page_dmp (in dmp/) This module provides an implementation of several types of DMPs. It depends on both the DynamicalSystems and FunctionApproximators modules, but no other.

\li \ref page_bbo (in bbo/) This module provides implementations of several stochastic optimization algorithms for the optimization of black-box cost functions. This module is completely independent of all other modules.  

\li \ref page_dmp_bbo (in dmp_bbo/) This module applies black-box optimization to the parameters of a DMP. It depends on all the other modules.

Each of the pages linked to above contains two sections: 

\li A tutorial that treats the concepts that are implemented 
\li A description of how these concepts have been implemented, and why it has been done so in this fashion.

A part of the functionality of the C++ code has been mirrored in Python. The Python version is probably the better language for getting to know dmpbbo (especially if you do not know C++ ;-) The C++ code is the better choice if you want to run dmpbbo on a real robot in a real-time environment. For now the Python code has not been documented well, please navigate the C++ documentation instead (class/function names have been kept consistent).

If you want a deeper understanding of the entire library, I recommend you to go through the pages in the order above. If you want to start coding immediately, I suggest to look at the \ref Demos (located in the demos/ and demos_python/ directories) to see how the functionality of the library may be used.

Some general considerations on the design of the library are here \ref page_design

\section sec_research_background Research background

In 2014, I decided to write one library that integrates the different research threads on the acquisition and optimization that I had been pursuing since 2009. These threads are listed below. Also, I wanted to provide a tutorial on dynamical movement primitives for students, along with code to try DMPs out in practice.

\li Representation and training of parameterized skills, i.e. motion primitives that adapt their trajectory to task parameters \cite matsubara11learning \cite silva12learning \cite stulp13learning.

\li Representing and optimizing gain schedules and force profiles as part of a DMP \cite buchli11learning \cite kalakrishnan11learning,


\li Showing that evolution strategies outperform reinforcement learning algorithms when optimizing the parameters of a DMP \cite stulp13robot, \cite stulp12policy_hal,

\li Demonstrating the advantages of using covariance matrix adaptation for the policy improvement \cite stulp12path \cite stulp12adaptive \cite stulp14simultaneous 

\li Using the same unified model for the model parameters of different function approximators \cite stulp15many. In fact, coding this library lead to this article, rather than vice versa.



*/

/** \page page_design Design Rationale

This page explains the overal design rationale for DmpBbo

\section sec_remarks General Remarks

\li Code legibility is more important to me than absolute execution speed (except for those parts of the code likely to be called in a time-critical context) or using all of the design patterns known to man (that is why I do not use PIMPL; it is not so legible for the uninitiated user. Also, I do not use the factory design pattern, but rather have clone() functions in classes ).

\li I learned to use Eigen whilst coding this project (learning-by-doing). So especially the parts I coded first might have some convoluted solutions (I didn't learn about Eigen::Ref til later...). Any suggestions for making the code more legible or efficient are welcome. The same goes for Python actually. So be gentle on me on this one; I myself will probably look back at this Python code in a few years and think: "How cute... I was just a Python baby when I coded that."

\li For consistency the names of Python modules (e.g. distribution_gaussian.py) have been kept consistent with their respective C++ implementation (e.g. DistributionGaussian.cpp). Several functions could also have been made mor Pythonic by exploiting duck-typing, but again, this has not always been done to keep consistency with the C++ code (where duck-typing is not possible).

\li For the organization of the code (directory structure), I went with this suggestion: http://stackoverflow.com/questions/13521618/c-project-organisation-with-gtest-cmake-and-doxygen/13522826#13522826

\li In function signatures, inputs come first (if they are references, they are const) and then outputs (if they are not const, they are inputs for sure). Exception: if input arguments have default values, they can come after outputs. Virtual functions should not have default function arguments (this is confusing in the derived classes). If they really need them, then you have to make different functions with different argument lists (see for example DmpContextual::train(), there are 6 of them for this reason).

\section sec_naming Naming convention

I mainly follow the following naming style: https://google.github.io/styleguide/cppguide.html#Naming

Notes:
\li Members end with a _, i.e. <code>this_is_a_member_</code>. (Exception: members in a POD (plain old data) class, which are public, and can be accessed directly)
\li I also use this convention: https://google.github.io/styleguide/cppguide.html#Access_Control
\li Abbreviation is the root of all evil! Long variable names are meaningful, and thus beautiful.

Exceptions to the style guide above:
\li functions start with low caps (as in Java, to distinguish them from classes) 
\li filenames for classes follow the classname (i.e. CamelCased)

The PEP Python naming conventions have been followed as much as possible, except for functions, which are camelCased, for consistency with the C++ code. 

\section Serialization

See \ref page_serialization
*/

/** \page page_todo Todo

\todo Explain why C++ implementation of bbo and dmp_bbo have only limited use.

\todo Documentation: Write a related pages with a table on which functionality is implemented in Python/Cpp

\todo Documentation: document Python classes/functions

\todo Documentation: Update documentation for parallel (No need for parallel in python, because only decay has been implemented for now)

\todo Plotting: setColor on ellipses?

\todo delay_cost in C++ not the same as in Python. Take the mean (as in Python) rather than the sum.

\todo Check documentation of dmp_bbo_robot


\todo demoOptimizationTaskWrapper.py: should there be a Task there also?
\todo clean up demoImitationAndOptimization
\todo clean up demoOptimizationDmpParallel: remove deprecated, only covar updates matter, make a flag
\todo FunctionApproximator::saveGridData in Python also
\todo further compare scripts
\todo testTrainingCompareCppPython.py => move part of it into demos/python/functionapproximators

\todo Table showing which functionality is available in Python/C++


 */ 

/** \defgroup Demos Demos
 */
 
/** \page page_demos Demos
 * 
 * DmpBbo comes with several demos.
 * 
 * The C++ demos are located in the dmpbbo/demos/ directory. Please see the README.md files located there.
 *
 * Many of the compiled executables are accompanied by a Python wrapper, which calls the executable,  and reads the files it writes, and then plots them (yes, I know about Python bindings; this approach allows better debugging of the format of the output files, which should always remain compatible between the C++ and Python versions of DmpBbo). For completeness, the pure Python demos are located in dmpbbo/demos_python.
 *
 Please note that this doxygen documentation only documents the C++ API of the libraries (in src/), not the demos. For explanations of the demos, please see the md files in the dmp/demos/ directory. 
 */

/** Namespace used for all classes in the project.
 */
namespace DmpBBO 
{
}

