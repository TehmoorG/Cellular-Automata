"""Implementations of Lorenz 96 and Conway's
Game of Life on various meshes"""

import numpy as np
import matplotlib.pyplot as plt


def lorenz96(initial_state, nsteps, constants=(1/101, 100, 8)):
    """
    Perform iterations of the Lorenz 96 model.

    Parameters
    ----------
    initial_state : array_like
        The initial state of the system as an iterable of floats.
    nsteps : int
        The number of iterations to perform.
    constants : tuple, optional
        The parameters alpha, beta, and gamma of the model as a tuple.

    Returns
    -------
    numpy.ndarray
        The state of the system after nsteps iterations.
    """
    alpha, beta, gamma = constants
    state = np.array(initial_state, dtype=float)
    for _ in range(nsteps):
        next_state = np.array(alpha * ((beta * state)
                                       - (np.roll(state, -2)
                                          - np.roll(state, 1))
                                       * np.roll(state, -1) + gamma))
        state = np.array(next_state)

    return state


def life(initial_state, nsteps, rules='basic', periodic=False):
    if rules == 'basic':
        return life_basic(initial_state, nsteps, periodic)
    elif rules == '2colour':
        return life2colour(initial_state, nsteps, periodic)
    elif rules == '3d':
        return life3d(initial_state, nsteps, periodic)
    else:
        raise Exception("Rules combination not recognised.")


def life_basic(initial_state, nsteps, periodic):
    """
    Perform iterations of Conway's Game of Life.

    Parameters
    ----------
    initial_state : array_like or list of lists
         Initial 2d state of grid in an array of booleans.
    nt : int
         Number of steps of Life to perform.

    Returns
    -------
    ndarray
         Final state of grid in array of booleans
    """
    state = np.array(initial_state)

    # main loop
    for _ in range(nsteps):
        nc = neighbour_count(np.array(state, bool), periodic)
        state = state * (nc == 2) + (nc == 3)

    return state


def life2colour(initial_state, nsteps, periodic=False):
    """
    Perform iterations of Conway's Game of Life.

    Parameters
    ----------
    initial_state : array_like or list of lists
         Initial 2d state of grid in an array of booleans.
    nsteps : int
         Number of steps of Life to perform.

    Returns
    -------
    ndarray
         Final state of grid in array of booleans
    """
    X2c = np.array(initial_state)

    # main loop
    for _ in range(nsteps):
        X2c_bin = np.square(X2c)
        nc_bin = neighbour_count(np.array(X2c_bin, bool), periodic)
        nc = neighbour_count(np.array(X2c), periodic)

        X2c = X2c * ((nc_bin == 2) + (nc_bin == 3))  # nc count 2,3 -> survives
        #  the following takes care of births
        X2c += (nc_bin == 3) * (X2c == 0) * ((nc == 3) +
                                             (nc == 1)) * (1)  # 1>-1
        X2c += (nc_bin == 3) * (X2c == 0) * ((nc == -3) +
                                             (nc == -1)) * (-1)  # -1>1
    return X2c


def life3d(initial_state, nsteps, periodic):
    state = np.array(initial_state)

    # main loop
    for _ in range(nsteps):
        nc = neighbour_count(np.array(state, bool), periodic)
        # Cells survive with 5 or 6 neighbours, cells turn on with 4 neighbours
        state = state * ((nc == 5) + (nc == 6)) + (nc == 4) - state * (nc == 4)

    return state


def neighbour_count(X, periodic=False):
    """
    Count the number of live neighbours of a regular n-d array.
    Parameters
    ----------

    X : numpy.ndarray of bools
       The Game of Life state.
    periodic : bool
       Whether boundaries are periodic
    Returns
    ------
    ndarray of ints
        Number of living neighbours for each cell.

    The algorithm looks in each direction, and increases the count in a cell
    if that neighbour is alive. This version is generic and will perform
    the count for any system dimension.

    """
    nx = X.shape
    neighbour_count = np.zeros(nx)
    
    combos = list(itertools.product(*(X.ndim * [(-1, 0, 1)])))

    # Because it will include every combination, it also has the "do nothing"
    # case to count the cell itself. We don't want that, so we remove
    # the all zeros entry from the list.
    combos.remove(X.ndim * (0, ))

    if periodic:
        # In the periodic case, we can use the np.roll function to shift
        # things.
        for combo in combos:
            neighbour_count[...] += 1 * np.roll(X, combo, range(X.ndim))
    else:
        # In the non periodic case, we loop over the combinations and
        # deal only with the slices which are actually relevant
        # e.g. in 1D we want to do
        # neighbour_count[0:nx-1] += X[1:nx]
        # neighbour_count[1:nx] += X[0:nx-1]

        SLICES = (slice(None, -1), slice(None, None), slice(1, None))

        def lhs_slice(combo):
            """Return the slice of the neighbour_count mesh to increment."""
            return tuple(SLICES[c + 1] for c in combo)

        def rhs_slice(combo):
            """Return the slice of the X mesh we're testing."""
            return tuple(SLICES[1 - c] for c in combo)

        for combo in combos:
            neighbour_count[lhs_slice(combo)] += 1 * X[rhs_slice(combo)]

    return neighbour_count




# The routines below are plotting aids. They do not need to modified and should not be called
# in the final versions of your functions.


def plot_lorenz96(data, label=None):
    """
    Plot 1d array on a circle

    Parameters
    ----------
    data: arraylike
        values to be plotted
    label:
        optional label for legend.
    """

    offset = 8

    data = np.asarray(data)
    theta = 2 * np.pi * np.arange(len(data)) / len(data)

    vector = np.empty((len(data), 2))
    vector[:, 0] = (data + offset) * np.sin(theta)
    vector[:, 1] = (data + offset) * np.cos(theta)

    theta = np.linspace(0, 2 * np.pi)

    rings = np.arange(int(np.floor(min(data)) - 1), int(np.ceil(max(data))) + 2)
    for ring in rings:
        plt.plot((ring + offset) * np.cos(theta), (ring + offset) * np.sin(theta), "k:")

    fig_ax = plt.gca()
    fig_ax.spines["left"].set_position(("data", 0.0))
    fig_ax.spines["bottom"].set_position(("data", 0.0))
    fig_ax.spines["right"].set_color("none")
    fig_ax.spines["top"].set_color("none")
    plt.xticks([])
    plt.yticks(rings + offset, rings)
    plt.fill(vector[:, 0], vector[:, 1], label=label, fill=False)
    plt.scatter(vector[:, 0], vector[:, 1], 20)


def plot_array(data, show_axis=False, cmap=plt.cm.get_cmap("seismic"), **kwargs):
    """Plot a 1D/2D array in an appropriate format.

    Mostly just a naive wrapper around pcolormesh.

    Parameters
    ----------

    data : array_like
        array to plot
    show_axis: bool, optional
        show axis numbers if true
    cmap : pyplot.colormap or str
        colormap

    Other Parameters
    ----------------

    **kwargs
        Additional arguments passed straight to pyplot.pcolormesh
    """
    plt.pcolormesh(
        1 * data[-1::-1, :], edgecolor="y", vmin=-2, vmax=2, cmap=cmap, **kwargs
    )

    plt.axis("equal")
    if show_axis:
        plt.axis("on")
    else:
        plt.axis("off")
