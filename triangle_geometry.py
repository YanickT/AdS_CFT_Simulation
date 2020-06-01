import pygame as pg
import numpy as np
import random
import time
from numba import njit, prange

COLORS = {1: (255, 0, 0), -1: (0, 0, 255), 0: (255, 255, 255)}


def create_field(size, d_state=-1):
    """
    Creates a field with height N.
    States S in a row r are given by: S(r) = r + 2.
    Disables states are filled with np.NaN.
    Therefore, field.shape == (size, size + 2)
    :param size: int = height of the field
    :param d_state: {-1, 1} = default field state
    :return: np.array, shape == (size, size + 2) = State field of the Ising model
    """
    field = np.zeros((size, size + 2))
    field.fill(0)
    for y, row in enumerate(field):
        for x in range(y + 2):
            row[x] = d_state
    return field


def prepare_states(field, mgfield):
    """
    Set every state to a random state determined by mgfield.
    :param field: 2d array: state field of the Ising model
    :param mgfield: float: default magnetic field
    :return: void
    """
    height = field.shape[0]
    for y in range(height):
        for x in range(y + 2):
            if random.random() < (mgfield + 1) / 2:
                field[y, x] = 1


def set_boundary(field, angle, black_hole_spin):
    """
    Force the given boundary conditions to the given field.
    :param field: np.array (2d) = state field of the Ising model
    :param angle: in [0, 2*pi] = Angle of the entangelment
    :param black_hole_spin: {-1, 1} = Spin orientation of the centred balckhole
    :return: void
    """
    ratio = angle / (2 * np.pi)

    field[0, 0] = black_hole_spin
    field[0, 1] = black_hole_spin

    for x in range(1 + field.shape[0]):
        if abs((2 * x + 1) / field.shape[1] - 1) < ratio:
            field[-1, x] = -1 * black_hole_spin
        else:
            field[-1, x] = black_hole_spin


def get_couplings(shape, cur_rad, mass):
    """
    Function calculates the coupling constants for each position for its neighbours.
    :return: np.array of shape (N, N, 3). array[y, x, (horizontal | up | down)]
    """
    js = np.zeros((shape, shape, 3))
    pref_jh = np.pi * cur_rad / (2 * shape)
    pref_jv = 2 * np.pi * cur_rad * np.sqrt(mass) / shape
    pi_2n = np.pi / (shape * 2)

    for y in range(shape):
        j_horizontal = pref_jh / np.cos(pi_2n * (y + 0.5))
        for x in range(shape):
            js[y, x, 0] = j_horizontal
            js[y, x, 1] = pref_jv / np.cos(pi_2n * y + pi_2n)
            js[y, x, 2] = pref_jv / np.cos(pi_2n * y)

            # normalize to max_field
            js[y, x, :] /= 2 * np.sum(js[y, x, :])

    return js


def prepare_pygame(width, height, size, blocksize=5):
    """
    Sets the pygame widget and the necessary configurations up. Futhermore creates a grid for the Ising states.
    :param width: int = total width of the widget
    :param height: int = total height of the widget
    :param size: int = number of states in the square field
    :param blocksize: int = length of edge of single state square in field
    :return: List[List[(pg.Surface, pg.Rect)]] = Grid of cells (for each state one)
    """
    pg.init()
    pg.display.set_caption('Ryu-Takayanagi Ising Simulator using Pygame')
    display = pg.display.set_mode((width, height))

    cell_grid = [[0 for i in range(size + 2)] for j in range(size)]
    for y in range(size):
        x_0 = (size - y) / 2
        for x in range(y + 2):
            # create a surface for each rectangle (this will only be updated if necessary)
            area = pg.Surface((blocksize, blocksize)).convert()
            area.fill((0, 0, 255))
            cell_grid[y][x] = (area, area.get_rect())
            cell_grid[y][x][1].center = ((x + x_0) * blocksize, y * blocksize)
            display.blit(cell_grid[y][x][0], cell_grid[y][x][1])

    return display, cell_grid


@njit(parallel=True)
def update_field(states, js, beta, loops):
    """
    Update loops random selected states in the Ising model. Performance boost using jit-numba with parallelization of
    the operations done by one loop.
    :param states: np.2darray = Field containing the ising model states
    :param js: np.3darray = 3d Array with shape [N, N, 3], N = states.shape[0], contains the coupling constants
    :param beta: float = inverse temperature for the ising model
    :param loops: int = number of states to check
    :return: void
    """
    size = states.shape[0]
    for i in prange(loops):
        y = int(random.random() * (size - 2) + 1)
        n_x = y + 2
        x = int(random.random() * n_x)
        x_add = (x + 1) % n_x
        x_sub = (x - 1 + n_x) % n_x

        h = js[y, x, 0] * (states[y, x_add] + states[y, x_sub]) + \
            js[y, x, 1] * (states[y + 1, x] + states[y + 1, x_add]) + \
            js[y, x, 2] * (states[y - 1, x] + states[y - 1, x_sub])
        states[y, x] = 1 if random.random() < 1 / (1 + np.exp(-2 * beta * h)) else -1


def update_display(cell_grid, result, display_field, display):
    """
    Updates the GUI. Especially the rectangles of the states that changed (given by result).
    :param cell_grid: List[List[(pg.Surface, pg.Rect)]] = List of pygame surfaces (representing states) and their rect.
    :param result: [(y, x), ...] = List of tuples containing the positions of the states that have changed.
    :param display_field: np.2darray = 2D Array of the states in the ising model
    :param display: pg.display = default display of pygame
    :return: void
    """
    for pos in np.argwhere(result):
        cell_grid[pos[0]][pos[1]][0].fill(COLORS[display_field[pos[0], pos[1]]])
        display.blit(cell_grid[pos[0]][pos[1]][0], cell_grid[pos[0]][pos[1]][1])


def main(width, height, n, angle, bhs, mass, cur_rad, weight, beta, loops, blocksize, times):
    """
    Main function starting the whole simulation (Ising model).
    :param width: int = width of the widget
    :param height: int = height of the widget
    :param n: int = number of states in the triangular field
    :param angle: float in [0, 2 * np.pi] = represents connection between the seperated regions
    :param bhs: int in {-1, 1} = state of the black in the center
    :param mass: float in (0, 1] = mass of the black hole
    :param cur_rad: float = curvature of the AdS space
    :param weight: flaot = weight of one iteration (for low-pass filter [suppresses fluctuations]). Set to 1 to
                           deactivate the filter
    :param beta: float = inverse temperature in the model
    :param loops: int = number of states to check in one sub-iteration
    :param blocksize: int = size of edge of single state square
    :param times: int = number of sub-iterations per iteration (iterations before displaying)
    :return: void
    """

    # create field an prepare
    states = create_field(n)
    prepare_states(states, -0.7)
    set_boundary(states, angle, bhs)

    avg_states = np.copy(states)
    js = get_couplings(n, cur_rad, mass)

    display, cell_grid = prepare_pygame(width, height, n, blocksize)
    for row in cell_grid:
        print([1 if e != 0 else 0 for e in row])
    display_field = np.where(avg_states > 0, 1, (np.where(avg_states < 0, -1, 0)))
    update_display(cell_grid, display_field != 0, display_field, display)
    old_display_field = np.copy(display_field)

    run = True
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        t1 = time.time()
        for i in range(times):
            update_field(states, js, beta, loops)
            avg_states = (1 - weight) * avg_states + weight * states
        print(time.time() - t1)

        display_field = np.where(avg_states > 0, 1, (np.where(avg_states < 0, -1, 0)))
        result = display_field != old_display_field
        old_display_field = np.copy(display_field)

        update_display(cell_grid, result, display_field, display)
        pg.display.update()


if __name__ == "__main__":
    blocksize = 2
    n = 10
    width, height = (n + 1) * blocksize, n * blocksize
    angle = 0.6 * np.pi
    mass = 1
    cur_rad = 1
    weight = 0.01
    loops = int(n / 2 * (n - 1) * 10)
    beta = 10
    bhs = -1
    times = 30
    main(width, height, n, angle, bhs, mass, cur_rad, weight, beta, loops, blocksize, times)
