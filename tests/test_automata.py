"""
Test automata module functions
Note: To ensure automata can be imported successfully, we recommend setting your PYTHONPATH prior to running this test file. This can be done in a terminal via, e.g.
export PYTHONPATH=/path_to/mpm-assessment-1
or similar.
"""
import os
import numpy as np
import automata
import pytest

BASE_PATH = os.path.dirname(__file__)


class TestLorenz96(object):
    """Class for testing the lorenz96 automata"""

    def test_lorenz96_basic(self):
        """Test Lorenz 96 implementation using pre-defined data"""
        initial64 = np.load(os.sep.join((BASE_PATH, "lorenz96_64_init.npy")))
        onestep64 = np.load(os.sep.join((BASE_PATH, "lorenz96_64_onestep.npy")))
        assert np.isclose(automata.lorenz96(initial64, 1), onestep64).all()
        thirtystep64 = np.load(os.sep.join((BASE_PATH, "lorenz96_64_thirtystep.npy")))
        assert np.isclose(automata.lorenz96(initial64, 30), thirtystep64).all()

    def test_steady_state(self):
        """Test Lorenz96 when all elements are 8. This should not change"""
        initial_state = np.full((64,), 8.0)  # An array of 64 eights
        result = automata.lorenz96(initial_state, 1)
        # Set a higher tolerance due to possible floating point errors
        assert np.isclose(result, 8.0, atol=1e-6).all(), "Failed steady state test"

    def test_forcing_increase(self):
        initial_state = np.full((64,), 7.0)
        result = automata.lorenz96(
            initial_state, 100
        )  # Assuming 100 steps will show the change
        assert (result > 7.0).all(), "Failed forcing increase test"

    def test_lorenz96_one_different(self):
        initial_state = np.full(64, 8.0)  # An array of 64 elements, all equal to 8
        i = 3  # Select an arbitrary index for changing the value
        initial_state[i] = 9.0  # Change the i-th value to 9

        # Apply your Lorenz 96 function
        result = automata.lorenz96(initial_state, nsteps=1, constants=(1 / 101, 100, 8))

        # Expected values
        expected_i_minus_1 = 800 / 101
        expected_i = 908 / 101
        expected_i_plus_1 = 816 / 101

        # Assert the expected values
        assert np.isclose(result[i - 1], expected_i_minus_1, atol=1e-6)
        assert np.isclose(result[i], expected_i, atol=1e-6)
        assert np.isclose(result[i + 2], expected_i_plus_1, atol=1e-6)

        # Assert that all other values remain approximately 8.0
        assert np.isclose(
            result[
                (result != expected_i_minus_1)
                & (result != expected_i)
                & (result != expected_i_plus_1)
            ],
            8.0,
            atol=1e-6,
        ).all()

    def test_different_constants(self):
        # testing with different α, β, γ values
        initial_state = [8.0, 8.0, 8.0, 8.0, 8.0]
        constants = (1 / 70, 69, 2)  # new constants
        # Hand Calculated
        expected = [
            7.914285714285714,
            7.914285714285714,
            7.914285714285714,
            7.914285714285714,
            7.914285714285714,
        ]
        result = automata.lorenz96(initial_state, 1, constants=constants)
        assert np.isclose(result, expected).all()


class TestLife:
    @pytest.mark.parametrize(
        "initial_state, n_steps, expected_state, rules, periodic",
        [
            # 2D Basic Rules
            (
                np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]]),
                1,
                np.array([[0, 1, 1], [0, 0, 0], [1, 0, 1]]),
                "basic",
                False,
            ),
            # 2D Two-Colour Rules
            (
                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 2, 0], [0, 1, 0, 0]]),
                1,
                np.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 2, 0], [0, 1, 1, 0]]),
                "2colour",
                False,
            ),
            # Periodic Conditions
            (
                np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
                1,
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                "basic",
                True,
            ),
        ],
    )
    def test_life(self, initial_state, n_steps, expected_state, rules, periodic):
        result = automata.life(initial_state, n_steps, rules, periodic)
        np.testing.assert_array_equal(result, expected_state)

    def test_3d_conditions(self):
        # This tests the scenario for a 3D grid.
        initial_state = np.zeros((3, 3, 3))
        initial_state[1, 1, 1] = 1
        # Based on the 3D rules, the central cell should stay alive
        # with 0 or 1 neighbors, or die with more neighbors.
        expected_state = np.zeros((3, 3, 3))
        result = automata.life(initial_state, 1, rules="3d", periodic=False)
        np.testing.assert_array_equal(result, expected_state)

    def test_non_square_grid(self):
        initial_state = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]])
        expected_state = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
        result = automata.life(initial_state, 1, rules="basic", periodic=False)
        np.testing.assert_array_equal(result, expected_state)
