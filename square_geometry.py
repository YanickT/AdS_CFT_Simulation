import pygame as pg
import numpy as np
import random
import time
from numba import njit, prange

COLORS = {1: (255, 0, 0), -1: (0, 0, 255), 0: (255, 255, 255)}


def prepare_states(field, mgfield):
    """
    Set every state to a random state determined by mgfield.
    :param field: 2d array: state field of the Ising model
    :param mgfield: float: default magnetic field
    :return: void
    """
    field.fill(-1)
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            if random.random() < (mgfield + 1) / 2:
                field[y, x] = 1


def set_boundary(field, angle, black_hole_spin):
    """
    Force the given boundary conditions to the given field.
    :param field: np.array of shape (N, N) = state field of the Ising model
    :param angle: in [0, 2*pi] = Angle of the entangelment
    :param black_hole_spin: in {-1, 1} = Spin orientation of the centred balckhole
    :return: void
    """
    ratio = angle / (2 * np.pi)
    for i in range(field.shape[0]):
        field[0][i] = black_hole_spin
        if abs((2 * i + 1) / field.shape[0] - 1) < ratio:
            field[-1][i] = -1 * black_hole_spin
        else:
            field[-1][i] = black_hole_spin


def get_couplings(shape, cur_rad, mass):
    """
    Function calculates the coupling constants for each position for its neighbours.
    :return: np.array of shape (N, N, 3). array[y, x, (horizontal | up | down)]
    """
    js = np.zeros((shape, shape, 3))

    sec = lambda y: 1 / np.cos((np.pi / 2) * (2 * y + 1) / (2 * shape))
    j_hor = lambda y: np.pi * cur_rad / (2 * shape) * sec(y)
    j_ver = lambda y: 2 * np.pi * np.sqrt(mass) * cur_rad / shape * sec(y)

    for y in range(shape):
        j_side = j_hor(y)
        j_up = j_ver(y)
        j_down = j_ver(y - 1)

        # normalize coupling
        norm = 2 * j_side + j_up + j_down
        j_side /= norm
        j_up /= norm
        j_down /= norm

        for x in range(shape):
            js[y, x, 0] = j_side
            js[y, x, 1] = j_up
            js[y, x, 2] = j_down

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

    cell_grid = []
    for y in range(size):
        temp = []
        for x in range(size):
            # create a surface for each rectangle (this will only be updated if necessary)
            area = pg.Surface((blocksize, blocksize)).convert()
            area.fill((0, 0, 255))
            temp.append((area, area.get_rect()))
            temp[-1][1].center = (x * blocksize, y * blocksize)
            display.blit(temp[-1][0], temp[-1][1])
        cell_grid.append(temp)

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
        x, y = int(random.random() * size), int(random.random() * (size - 2) + 1)
        h = js[y, x, 0] * states[y, (x + 1) % size] + js[y, x, 0] * states[y, (x + size - 1) % size] + \
            js[y, x, 1] * states[y + 1, x] + js[y, x, 2] * states[y - 1, x]
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
    :param n: int = number of states in the square field
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
    states = np.zeros((n, n), dtype=np.float32)
    prepare_states(states, -0.7)
    set_boundary(states, angle, bhs)

    avg_states = np.copy(states)
    js = get_couplings(n, cur_rad, mass)

    display, cell_grid = prepare_pygame(width, height, n, blocksize)
    display_field = np.where(avg_states > 0, 1, -1)
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
        print(time.time()-t1)
        display_field = np.where(avg_states > 0, 1, -1)
        result = display_field == old_display_field
        old_display_field = np.copy(display_field)

        update_display(cell_grid, result, display_field, display)
        pg.display.update()

    np.save("data.csv", display_field)
    pg.quit()


if __name__ == "__main__":
    blocksize = 4
    n = 125
    width, height = n * blocksize, n * blocksize
    angle = 0.6 * np.pi
    mass = 1
    cur_rad = 1
    weight = 0.01
    loops = n * (n - 1) * 10
    beta = 10
    bhs = -1
    times = 5
    main(width, height, n, angle, bhs, mass, cur_rad, weight, beta, loops, blocksize, times)