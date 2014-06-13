# import numpy
# import sys
# import os

import _dmpbbo


class DmpBbo(object):
    def __init__(self, **kwargs):
        self._delegate = _dmpbbo.DmpBbo()

    def run(self, tau, n_time_steps, n_basis_functions, input_dim, intersection, save_dir):
        self._delegate.run(tau, n_time_steps, n_basis_functions, input_dim, intersection, save_dir)
        # return numpy.array(self._delegate.produce(list(params)))

#if __name__ == '__main__':
    #p = Praat()

    #for _ in range(100):
        #l1 = list(numpy.random.randn(i.d))
        #l2 = list(numpy.random.randn(i.D))

        #i.update(l1, l2)

    #print i.predict(range(7))
    #print i.predict(range(7))
    #print i.predict(range(7))

    #print i.predict_inverse([4, 5, 6])

