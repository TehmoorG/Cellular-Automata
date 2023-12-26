import numpy as np
import pytest
import os
import sys


base_dir = os.environ.get("BASEDIR", os.path.dirname(__file__) + os.sep)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGeneral(object):
    # Test 1: Some general tests
    def test_flake8_an(self):
        from flake8.api import legacy as flake8
        import automata

        style_guide = flake8.get_style_guide()
        result = style_guide.input_file(
            os.sep.join((os.environ.get("PYTHONPATH", "./"), "automata.py"))
        )
        assert result.total_errors == 0
        assert hasattr(automata, "lorenz96")
        assert hasattr(automata, "life")


class TestLorenz(object):
    # Test lorenz96
    # Test 2
    def test_lorenz96_stabiity(self):
        import automata

        X = np.full(16, 8.0)
        X = automata.lorenz96(X, 16)
        Z = np.full(16, 8.0)
        assert (X == Z).all()

    # Test 3
    def test_lorenz_ints(self):
        import automata

        X = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
        Y = np.array([2.95, 3.8, 4.71, 5.61, 6.83, 8.12, 7.37, 6.34, 5.3, 4.2, 3.1])
        X = automata.lorenz96(X, 2)
        assert np.allclose(X, Y, atol=0.1, rtol=0.1)


class TestLife(object):
    # Tests on 2D Game of Life
    # Test 4
    def test_life_runs_on_array(self):
        import automata

        X = np.zeros((3, 3), bool)
        X[1, :] = True
        X = automata.life(X, 1)

        Z = np.array(((False, True, False), (False, True, False), (False, True, False)))

        assert (np.array(X) == np.array(Z)).all()

    # Test 5
    def test_life_runs_on_arraylikes(self):
        import automata

        X = ((False, False, False), (True, True, True), (False, False, False))

        X = automata.life(X, 1)

        Z = np.array(((False, True, False), (False, True, False), (False, True, False)))

        assert (np.array(X) == np.array(Z)).all()

    # Test 6
    def test_life_glider(self):
        import automata

        X = np.array(
            (
                (0, 0, 0, 0, 0),
                (0, 1, 1, 1, 0),
                (0, 0, 0, 1, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            bool,
        )

        X = automata.life(X, 4)

        Z = np.array(
            (
                (0, 0, 1, 1, 1),
                (0, 0, 0, 0, 1),
                (0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            bool,
        )

        assert (np.array(X) == np.array(Z)).all()

    # Test 7
    @pytest.mark.timeout(4)
    def test_life_on_data(self):
        import automata

        X = np.load(base_dir + "life_small_init.npy")
        Y = np.load(base_dir + "life_small_final.npy")

        assert np.array(automata.life(X, 10) == Y).all()

    # Test 8
    @pytest.mark.timeout(4)
    def test_life_large(self):
        import automata

        X = np.load(base_dir + "life_random_large_init.npy")
        Y = np.load(base_dir + "life_random_large_final.npy")
        X = automata.life(X, 32)
        assert (X == Y).all()


class TestLifePeriodic(object):
    # Tests for Periodic Game of Life
    # Test 9
    def test_life_runs_on_array_periodic(self):
        import automata

        X = np.zeros((4, 4), bool)
        X[1, :2] = True
        X[1, 3] = True
        X = automata.life(X, 1, periodic=True)

        Z = np.array(
            (
                (True, False, False, False),
                (True, False, False, False),
                (True, False, False, False),
                (False, False, False, False),
            )
        )

        assert (np.array(X) == np.array(Z)).all()

    # Test 10
    def test_life_glider_periodic(self):
        import automata

        X = np.array(
            (
                (0, 0, 0, 0, 0),
                (0, 1, 1, 1, 0),
                (0, 0, 0, 1, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            bool,
        )

        X0 = automata.life(X.copy(), 20, periodic=True)

        Z = np.array(
            (
                (0, 0, 0, 0, 0),
                (0, 1, 1, 1, 0),
                (0, 0, 0, 1, 0),
                (0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0),
            ),
            bool,
        )

        assert (np.array(X0) == np.array(Z)).all()

        X0 = automata.life(X.T.copy(), 20, periodic=True)
        assert (np.array(X0) == np.array(Z.T)).all()

        X0 = automata.life(X[::-1, :].copy(), 20, periodic=True)
        assert (np.array(X0) == np.array(Z[::-1, :])).all()

        X0 = automata.life(X[:, ::-1].copy(), 20, periodic=True)
        assert (np.array(X0) == np.array(Z[:, ::-1])).all()

    # Test 11
    @pytest.mark.timeout(2)
    def test_life_periodic_on_data(self):
        import automata

        X = np.load(base_dir + "life_small_init.npy")
        Y = np.load(base_dir + "life_periodic_small_final.npy")

        assert (np.array(automata.life(X, 10, periodic=True)) == Y).all()

    # Test 12
    @pytest.mark.timeout(4)
    def test_life_periodic_large(self):
        import automata

        X = np.load(base_dir + "lifep_random_large_init.npy")
        Y = np.load(base_dir + "lifep_random_large_final.npy")
        X = automata.life(X, 60, periodic=True)
        assert (X == Y).all()


class Test2ColourLife(object):
    # Tests on 2 Colour Game of Life
    # Test 13
    @pytest.mark.timeout(2)
    def test_life2colour_splitcol(self):
        import automata

        X = np.load(base_dir + "life2c_1_init.npy")
        Y = np.load(base_dir + "life2c_1_final.npy")
        X = automata.life(X, 17, rules="2colour", periodic=True)
        assert (X == Y).all()

    # Test 14
    @pytest.mark.timeout(2)
    def test_life2colour_rand_small(self):
        import automata

        X = np.load(base_dir + "life2c_2_init.npy")
        Y = np.load(base_dir + "life2c_2_final.npy")
        X = automata.life(X, 23, rules="2colour", periodic=True)
        assert (X == Y).all()

    # Test 15
    @pytest.mark.timeout(4)
    def test_life2colour_rand_large(self):
        import automata

        X = np.load(base_dir + "life2c_random_large_init.npy")
        Y = np.load(base_dir + "life2c_random_large_final.npy")
        X = automata.life(X, 24, rules="2colour", periodic=True)
        assert (X == Y).all()

    # Test 16
    @pytest.mark.timeout(4)
    def test_life2colour_rand_large_nonperiodic(self):
        import automata

        X = np.load(base_dir + "life2c_random_large_init_nonperiodic.npy")
        Y = np.load(base_dir + "life2c_random_large_final_nonperiodic.npy")
        X = automata.life(X, 36, rules="2colour", periodic=False)
        assert (X == Y).all()


class Test3DLife(object):
    # 3D game of life baby
    # Test 17
    def test_varied_initial(self):
        import automata

        X = np.array(
            [
                [
                    [1, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 1],
                    [1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0],
                    [1, 1, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 0, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1],
                ],
            ],
            dtype=int,
        )
        Y = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
            dtype=int,
        )

        assert (automata.life(X, 2, rules="3d") == Y).all()

    # Test 18
    def test_varied_initial_periodic(self):
        import automata

        X = np.array(
            [
                [
                    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                ],
                [
                    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                ],
                [
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                ],
            ],
            dtype=int,
        )

        Y = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                ],
                [
                    [1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    [1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                ],
                [
                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                ],
            ],
            dtype=int,
        )

        assert (automata.life(X, 100, rules="3d", periodic=True) == Y).all()

    # Test 19
    @pytest.mark.timeout(4)
    def test_life3d_rand_large(self):
        import automata

        X = np.load(base_dir + "life3d_random_large_init.npy")
        Y = np.load(base_dir + "life3d_random_large_final.npy")
        assert (automata.life(X, 24, rules="3d") == Y).all()

    # Test 20
    @pytest.mark.timeout(4)
    def test_life3d_rand_large_periodic(self):
        import automata

        X = np.load(base_dir + "life3d_random_large_init_periodic.npy")
        Y = np.load(base_dir + "life3d_random_large_final_periodic.npy")
        assert (automata.life(X, 40, rules="3d", periodic=True) == Y).all()
