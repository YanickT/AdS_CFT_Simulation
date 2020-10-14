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
    :return: np.array of shape (N, N, 5). array[y, x, (horizontal | up | up_right |  down | down_left)]
    """
    js = np.zeros((shape, shape + 2, 5))
    js.fill(np.NaN)

    n = lambda row: row + 2
    sec = lambda y: 1 / np.cos((np.pi / 2) * (2 * y + 1) / (2 * shape))

    # Hinrichsens method on square geoemetry
    j_hor = lambda y: np.pi * cur_rad / (2 * shape) * sec(y)
    j_ver = lambda y: 2 * np.pi * np.sqrt(mass) * cur_rad / n(y) * sec(y)

    # geometric coupling
    n_n = lambda r1, r2: n(r1) / n(r2)
    c = lambda i, j, r1, r2: int(abs(r1 - r2) == 1) * (
            max(i, n_n(r1, r2) * (j + 1))
            + max(i + 1, n_n(r1, r2) * j)
            - max(i, n_n(r1, r2) * j)
            - max(i + 1, n_n(r1, r2) * (j + 1)))

    for y in range(shape):
        j_side = j_hor(y)
        j_up = j_ver(y)
        j_down = j_ver(y - 1)

        # normalize the coupling
        norm = 2 * j_side + j_up + j_down
        j_side /= norm
        j_up /= norm
        j_down /= norm

        for x in range(n(y)):
            # horizontal
            js[y, x, 0] = j_side
            # up
            js[y, x, 1] = c(x, x, y, y + 1) * j_up
            # up right
            js[y, x, 2] = c(x, (x + 1) % n(y + 1), y, y + 1) * j_up

            # down
            js[y, x, 3] = c(x, x, y, y - 1) * j_down
            # down left
            js[y, x, 4] = c(x, (x - 1 + n(y - 1)) % n(y - 1), y, y - 1) * j_down

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
        # calculate the width of a block
        block_width = width // (y + 2)

        # create list of squares which are bigger
        larger = [int(round(value)) for value in np.linspace(0, y + 2, width % (y + 2))]


        pos_x = 0
        for x in range(y + 2):
            # value = int(255 / (y / 2 + 1) * x)
            # color = (value, value, value)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            width_ = block_width + int(x in larger)
            area = pg.Surface((width_, blocksize)).convert()
            area.fill(color)
            cell_grid[y][x] = (area, area.get_rect())
            cell_grid[y][x][1].center = (pos_x + width_ // 2, y * blocksize + blocksize // 2)
            pos_x += width_
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

        # (horizontal | up | up_right | down | down_left)
        h = js[y, x, 0] * (states[y, x_add] + states[y, x_sub]) + \
            js[y, x, 1] * states[y + 1, x] + \
            js[y, x, 2] * states[y + 1, x_add] + \
            js[y, x, 3] * states[y - 1, x] + \
            js[y, x, 4] * states[y - 1, x_sub]
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
    :param mass: float in (0, 1] = mass of the black hole * -1
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
    # while True:
    #    pg.display.update()

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

    pg.quit()


if __name__ == "__main__":
    blocksize = 4
    n = 100
    width, height = (n + 1) * blocksize, n * blocksize
    angle = 0.6 * np.pi
    mass = 1
    cur_rad = 1
    weight = 0.01
    loops = int(n / 2 * (n - 1) * 10)
    beta = 10
    bhs = -1
    times = 5
    main(width, height, n, angle, bhs, mass, cur_rad, weight, beta, loops, blocksize, times)