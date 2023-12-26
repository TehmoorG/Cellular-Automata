import numpy as np
import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestLorenz(object):
    # Test lorenz96
    # Test 2
    def test_lorenz96_stabiity(self):
        import automata

        X = np.full(16, 8.)
        X = automata.lorenz96(X, 16)
        Z = np.full(16, 8.)
        assert (X == Z).all()

    # Test 3
    def test_lorenz_ints(self):
        import automata
        X = np.array([3., 4., 5., 6., 7., 8., 7., 6., 5., 4., 3.])
        Y = np.array([2.95, 3.8, 4.71, 5.61, 6.83, 8.12, 7.37, 6.34, 5.3, 4.2, 3.1])
        X = automata.lorenz96(X, 2)
        assert np.allclose(X, Y, atol=0.1, rtol=0.1)