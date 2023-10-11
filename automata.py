"""Implementations of Lorenz 96 and Conway's
Game of Life on various meshes"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def lorenz96(initial_state, nsteps, constants=(1/101, 100, 8)):
    """
    Perform iterations of the Lorenz 96 update.

    Parameters
    ----------
    initial_state : array_like or list
        Initial state of lattice in an array of floats.
    nsteps : int
        Number of steps of Lorenz 96 to perform.

    Returns
    -------

    numpy.ndarray
         Final state of lattice in array of floats

    >>> x = lorenz96([8.0, 8.0, 8.0], 1)
    >>> print(x)
    array([8.0, 8.0, 8.0])
    """

    # write your code here to replace this return statement
    alpha, beta, gamma = constants
    state = np.array(initial_state, dtype=float)

    for _ in range(nsteps):
        new_state = np.zeros_like(state)
        N = len(state)

        for i in range(N):
            new_state[i] = alpha * (beta * state[i] + (state[(i - 2) % N] - state[(i + 1) % N]) * state[(i - 1) % N] + gamma)

        state = new_state

    return state


def life(initial_state, nsteps, rules="basic", periodic=False):
    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of ints.
    nsteps : int
        Number of steps of Life to perform.
    rules : str
        Choose a set of rules from "basic", "2colour" or "3d".
    periodic : bool
        If True, use periodic boundary conditions.
    Returns
    -------
    numpy.ndarray
         Final state of grid in array of ints.
    """

    # write your code here to replace return statement
    return NotImplemented


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
    theta = 2*np.pi*np.arange(len(data))/len(data)

    vector = np.empty((len(data), 2))
    vector[:, 0] = (data+offset)*np.sin(theta)
    vector[:, 1] = (data+offset)*np.cos(theta)

    theta = np.linspace(0, 2*np.pi)

    rings = np.arange(int(np.floor(min(data))-1),
                      int(np.ceil(max(data)))+2)
    for ring in rings:
        plt.plot((ring+offset)*np.cos(theta),
                 (ring+offset)*np.sin(theta), 'k:')

    fig_ax = plt.gca()
    fig_ax.spines['left'].set_position(('data', 0.0))
    fig_ax.spines['bottom'].set_position(('data', 0.0))
    fig_ax.spines['right'].set_color('none')
    fig_ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks(rings+offset, rings)
    plt.fill(vector[:, 0], vector[:, 1],
             label=label, fill=False)
    plt.scatter(vector[:, 0], vector[:, 1], 20)


def plot_array(data, show_axis=False,
               cmap=plt.cm.get_cmap('seismic'), **kwargs):
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
    plt.pcolormesh(1*data[-1::-1, :], edgecolor='y',
                   vmin=-2, vmax=2, cmap=cmap, **kwargs)

    plt.axis('equal')
    if show_axis:
        plt.axis('on')
    else:
        plt.axis('off')
