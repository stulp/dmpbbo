The integration tests compare the results of function approximator prediction and dynamical system integration (literally an "integration test") in Python and C++. If the resulting predictions/trajectories are very similar (max 10e-07 for any giving prediction or time step) the tests succeed.

The test script can also be called on the command line, e.g. `python3 test_dmp.py`. In this case, a visualization will be made.

Since the Python code calls exececutables generated from the C++ code, it is necessary to do `make install` in the root directory of dmpbbo before calling these integration tests/scripts.