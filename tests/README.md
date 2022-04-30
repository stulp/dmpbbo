The integration tests compare the results of function approximator prediction and dynamical system integration ("integration test" ;-) in Python and C++. If the resulting predictions/trajectories are very similar (max 10e-07 for any giving prediction or time step) the tests succeed.

For now there are no unit tests.