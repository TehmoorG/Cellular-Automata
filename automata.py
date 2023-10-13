"""Implementations of Lorenz 96 and Conway's
Game of Life on various meshes"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting tool


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
        Final state of lattice in an array of floats
    """

    alpha, beta, gamma = constants
    state = np.array(initial_state, dtype=float)
    N = len(initial_state)
    new_state = np.empty_like(state)  # Create a new state array

    for _ in range(nsteps):
        # Compute the first two elements
        new_state[0] = alpha * (beta * state[0] + (state[N - 2] - state[1]) * state[N - 1] + gamma)
        new_state[1] = alpha * (beta * state[1] + (state[N - 1] - state[2]) * state[0] + gamma)

        # Compute the elements between 2 and N-2
        new_state[2:N - 1] = alpha * (beta * state[2:N - 1] + (state[0:N - 3] - state[3:N]) * state[1:N - 2] + gamma)   

        # Compute the last element
        new_state[N - 1] = alpha * (beta * state[N - 1] + (state[N - 3] - state[0]) * state[N - 2] + gamma)

        # Update the state array
        state[:] = new_state

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
    state = np.array(initial_state, dtype=int)

    # Determine if we're working in 2D or 3D
    if rules == "3d":
        if state.ndim != 3:
            raise ValueError("Invalid grid dimension!")
        rows, cols, depth = state.shape
        for _ in range(nsteps):
            next_state = state.copy()
            
            for i in range(rows):
                for j in range(cols):
                    for k in range(depth):
                        total = 0  # Count the neighbors

                        for x in [-1, 0, 1]:
                            for y in [-1, 0, 1]:
                                for z in [-1, 0, 1]:
                                    if x == 0 and y == 0 and z == 0:
                                        continue  # Skip the current cell
                                    ni, nj, nk = i + x, j + y, k + z
                                    if periodic:  # Handle periodic boundary conditions
                                        ni %= rows
                                        nj %= cols
                                        nk %= depth
                                    elif ni < 0 or ni >= rows or nj < 0 or nj >= cols or nk < 0 or nk >= depth:
                                        continue  # Skip out of bounds

                                    total += state[ni, nj, nk]
                                    
                            if state[i, j, k] != 0:  # If cell is alive
                                if total < 5 or total > 6:  # Die
                                    next_state[i, j, k] = 0
                            else:  # Dead cell
                                if total == 4:
                                    next_state[i, j, k] = 1  # Birth
            
            state = next_state
        return state
    else:
        if state.ndim != 2:
            raise ValueError("Invalid grid dimension!")
        rows, cols = state.shape
        for _ in range(nsteps):
            next_state = state.copy()

            for i in range(rows):
                for j in range(cols):
                    total = 0
                    blue_neighbours = 0
                    red_neighbours = 0
                    for x in [-1, 0, 1]:
                        for y in [-1, 0, 1]:
                            if x == 0 and y == 0:
                                continue
                            ni, nj = i + x, j + y
                            if periodic:
                                ni %= rows
                                nj %= cols
                            elif ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                                continue

                            if state[ni][nj] > 0:
                                total += 1
                                if state[ni][nj] == 1:
                                    blue_neighbours += 1
                                else:
                                    red_neighbours += 1

                    if rules == "basic":
                        if state[i][j] == 1:  # If cell is alive
                            if total < 2 or total > 3:  # Die
                                next_state[i][j] = 0
                        else:  # Dead cell
                            if total == 3:
                                next_state[i][j] = 1

                    elif rules == "2colour":
                        if state[i][j] == 1 or state[i][j] == 2:  # If cell is alive
                            if total < 2 or total > 3:  # Die
                                next_state[i][j] = 0
                        else:  # Dead cell
                            if total == 3:
                                if blue_neighbours > red_neighbours:
                                    next_state[i][j] = 1
                                else:
                                    next_state[i][j] = 2

            state = next_state

        return state

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

