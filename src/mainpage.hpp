/**
 * @file mainpage.hpp
 * @brief File containing only documentation (for the Doxygen mainpage)
 * @author Freek Stulp
 */

/** Namespace used for all classes in the project.
 */
namespace DmpBBO {
}

/** \mainpage

\section sec_cui_bono What the doxygen documentation is for

This is the doxygen documentation of the C++ of the dmpbbo library. Its aim
is to document the C++ API.

A broader overview of the dmpbbo software, including the Python implementations,
can be found in the main README: https://github.com/stulp/dmpbbo

\section sec_overview_modules Overview of the modules/libraries

This library contains several modules for executing dynamical movement
primitives (DMPs). Each module has its own dedicated page.

\li \ref eigenutils (in eigenutils/) This header-only module provides utilities
for the Eigen matrix library (file IO, (de)serialization with JSON, and checking
whether code is real-time).

\li  \ref page_dyn_sys (in dynamicalsystems/) This module provides
implementations of several basic dynamical systems. DMPs are combinations of
such systems.

\li \ref page_func_approx (in functionapproximators/) This module provides
implementations of several function approximators. DMPs use function
approximators to learn and reproduce arbitrary smooth movements.

\li \ref page_dmp (in dmp/) This module provides an implementation of several
types of DMPs.


*/

/** \page page_design Design Rationale

This page explains the overal design rationale for DmpBbo

\section sec_remarks General Remarks

\li Code legibility is more important to me than absolute execution speed
(except for those parts of the code likely to be called in a time-critical
context) or using all of the design patterns known to man (that is why I do not
use PIMPL; it is not so legible for the uninitiated user.

\li I learned to use Eigen whilst coding this project (learning-by-doing). So
especially the parts I coded first might have some convoluted solutions (I
didn't learn about Eigen::Ref til later...). Any suggestions for making the code
more legible or efficient are welcome.

\li For the organization of the code (directory structure), I went with this
suggestion:
http://stackoverflow.com/questions/13521618/c-project-organisation-with-gtest-cmake-and-doxygen/13522826#13522826

\li In function signatures, inputs come first (if they are references, they are
const) and then outputs (if they are not const, they are inputs for sure).
Exception: if input arguments have default values, they can come after outputs.
Virtual functions should not have default function arguments (this is confusing
in the derived classes).

\section sec_naming Coding style

Formatting according to the Google style
(https://google.github.io/styleguide/cppguide.html) is done automatically with
`clang-format' The only difference to the Google formatting style is that the
opening bracket after a function header is on a newline, as this improves
legibility for me. See the `.clang-format` settings file in the root of the
repo.

I mainly follow the following naming style:
https://google.github.io/styleguide/cppguide.html#Naming

Notes:
\li Members end with a _, i.e. <code>this_is_a_member_</code>. (Exception:
members in a POD (plain old data) class, which are public, and can be accessed
directly) \li I also use this convention:
https://google.github.io/styleguide/cppguide.html#Access_Control \li
Abbreviation is the root of all evil! Long variable names are meaningful, and
thus beautiful.

Exceptions to the style guide above:
\li functions start with low caps (as in Java, to distinguish them from classes)
\li filenames for classes follow the classname (i.e. CamelCased)

The PEP Python naming conventions have been followed as much as possible, except
for functions, which are camelCased, for consistency with the C++ code.

*/

/** \defgroup Demos Demos
 */

/** \page page_demos Demos
 *
 * DmpBbo comes with several demos.
 *
 * The C++ demos are located in the dmpbbo/demos_cpp/ directory. Please see the
 README.md files located there.
 *
 * Many of the compiled executables are accompanied by a Python wrapper, which
 calls the executable,  and reads the files it writes, and then plots them (yes,
 I know about Python bindings; this approach allows better debugging of the
 format of the output files, which should always remain compatible between the
 C++ and Python versions of DmpBbo). For completeness, the pure Python demos are
 located in dmpbbo/demos_python.
 *
 Please note that this doxygen documentation only documents the C++ API of the
 libraries (in src/), not the demos. For explanations of the demos, please see
 the md files in the dmpbbo/demos_cpp/ directory.
 */

/** \page page_bibliography Bibliography

To ensure that all relevant entries are generated for the bibliography, here is
a list.

\cite buchli11learning
\cite ijspeert02movement
\cite ijspeert13dynamical
\cite kalakrishnan11learning
\cite kulvicius12joining
\cite matsubara11learning
\cite silva12learning
\cite stulp12adaptive
\cite stulp12path
\cite stulp12policy_hal
\cite stulp13learning
\cite stulp13robot
\cite stulp14simultaneous
\cite stulp15many




 */
