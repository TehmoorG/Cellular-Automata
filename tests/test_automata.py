"""
Test automata module functions
Note: To ensure automata can be imported successfully, we recommend setting your PYTHONPATH prior to running this test file. This can be done in a terminal via, e.g.
export PYTHONPATH=/path_to/mpm-assessment-1
or similar.
"""
import os
import numpy as np
import automata
BASE_PATH = os.path.dirname(__file__)

class TestLorenz96(object):
    """Class for testing the lorenz96 automata"""
    def test_lorenz96_basic(self):
        """Test Lorenz 96 implementation using pre-defined data"""
        initial64 = np.load(os.sep.join((BASE_PATH,
                                        'lorenz96_64_init.npy')))
        onestep64 = np.load(os.sep.join((BASE_PATH,
                                        'lorenz96_64_onestep.npy')))
        assert np.isclose(automata.lorenz96(initial64, 1), onestep64).all()
        thirtystep64 = np.load(os.sep.join((BASE_PATH,
                                        'lorenz96_64_thirtystep.npy')))
        assert np.isclose(automata.lorenz96(initial64, 30), thirtystep64).all()
