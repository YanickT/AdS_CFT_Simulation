import pygame as pg
import numpy as np
import random
import time
from numba import njit, prange


# colors for the simulation: 1: RED, -1: BLUE, 0: BLACK
COLORS = {1: (255, 0, 0), -1: (0, 0, 255), 0: (255, 255, 255)}


# time critical function. Use numba to speed things up
# Due to multiple calls with same input/output use cache
@njit(cache=True)
def fn(r):
    """
    Returns the number of fields based on their radial distance
    :param r: radial distance
    :return: number of fields at radial distance
    """
    return 4 * r + 4



# time critical function. Use numba to speed things up
# Due to multiple calls with same input/output use cache
@njit(cache=True)
def geo_couplings(x1, r1, r2):
    """
    Calculates the geometric couplings from x1, y1 to the y2 row
    :param x1: int = spherical position of the updating spin field
    :param r1: int = radius of the updating spin field
    :param r2: int = radius of the underlying spin fields
    :return: np.array, shape == (1, fn(r2)) = Couplings to [0, fn(r2)] in r2
    """
    n_n = lambda r1, r2: fn(r1) / fn(r2)
    c = lambda i, j, r1, r2: int(abs(r1 - r2) == 1) * (
            max(i, n_n(r1, r2) * (j + 1))
            + max(i + 1, n_n(r1, r2) * j)
            - max(i, n_n(r1, r2) * j)
            - max(i + 1, n_n(r1, r2) * (j + 1)))

    return np.array([c(x1, x2, r1, r2) for x2 in range(fn(r2))])


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
    field = np.zeros((size, fn(size)))
    field.fill(0)
    for y, row in enumerate(field):
        for x in range(fn(y)):
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
        for x in range(fn(y)):
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

    for i in range(fn(0)):
        field[0, i] = black_hole_spin

    for x in range(fn(field.shape[0] - 1)):
        if abs((2 * x + 1) / field.shape[1] - 1) < ratio:
            field[-1, x] = -1 * black_hole_spin
        else:
            field[-1, x] = black_hole_spin


def get_couplings(shape, cur_rad, mass):
    """
    Function calculates the coupling constants for each position for its neighbours.
    :return: np.array of shape (N, N, 3). array[y, x, (horizontal | up | down)]
    """
    js = np.zeros((shape, fn(shape), 3))
    js.fill(np.NaN)

    sec = lambda y: 1 / np.cos((np.pi / 2) * (2 * y + 1) / (2 * shape))

    # Hinrichsens method on square geoemetry
    j_hor = lambda y: np.pi * cur_rad / (2 * shape) * sec(y)
    j_ver = lambda y: 2 * np.pi * np.sqrt(mass) * cur_rad / fn(y) * sec(y)

    for y in range(1, shape):
        j_side = j_hor(y)
        j_up = j_ver(y)
        j_down = j_ver(y - 1)

        # normalize coupling
        norm = 2 * j_side + j_up + j_down
        j_side /= norm
        j_up /= norm
        j_down /= norm

        for x in range(fn(y)):
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
    # create GUI
    pg.init()
    pg.font.init()
    pg.display.set_caption('Ryu-Takayanagi Ising Simulator using Pygame')
    display = pg.display.set_mode((width, height))

    # create cells (for spins) in GUI
    # cell_grid is a 2D array with temp as line
    cell_grid = [[0 for i in range(fn(j))] for j in range(size)]
    for y in range(size):
        # calculate the width of a block
        block_width = width // fn(y)

        # create list of squares which are bigger
        larger = [int(round(value)) for value in np.linspace(0, fn(y), width % fn(y))]

        pos_x = 0
        for x in range(fn(y)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # <--------------
            # create a surface for each rectangle (this will only be updated if necessary)
            width_ = block_width + int(x in larger)
            area = pg.Surface((width_, blocksize)).convert()
            area.fill(color)
            cell_grid[y][x] = (area, area.get_rect())
            cell_grid[y][x][1].center = (pos_x + width_ // 2, y * blocksize + blocksize // 2)
            pos_x += width_
            display.blit(cell_grid[y][x][0], cell_grid[y][x][1])

    return display, cell_grid


# time critical function. Use numba to speed things up
@njit()
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
        n_x = fn(y)
        x = int(random.random() * n_x)

        h = js[y, x, 0] * (states[y, (x + 1) % n_x] + states[y, (x - 1 + n_x) % n_x]) + \
            np.sum(np.multiply(np.multiply(geo_couplings(x, y, y + 1), states[y + 1, :fn(y + 1)]), js[y, x, 1])) + \
            np.sum(np.multiply(np.multiply(geo_couplings(x, y, y - 1), js[y, x, 2]), states[y - 1, :fn(y - 1)]))

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

    # initialise coupling array
    js = get_couplings(n, cur_rad, mass)
    
    # copy stats for low-pass filter
    avg_states = np.copy(states)

    # initialise GUI objects
    display, cell_grid = prepare_pygame(width, height, n, blocksize)
    display_field = np.where(avg_states > 0, 1, (np.where(avg_states < 0, -1, 0)))
    update_display(cell_grid, display_field != 0, display_field, display)
    old_display_field = np.copy(display_field)

    # text fields for infomations
    time_txt = pg.font.SysFont('Courier New', 12)
    beta_txt = pg.font.SysFont('Courier New', 12)
    weight_txt = pg.font.SysFont('Courier New', 12)

    beta_img = beta_txt.render(f"beta: {'0' * (2 - len(str(beta))) + str(beta)}", False, (0, 0, 0), COLORS[-1])
    weight_img = weight_txt.render(f"weight: {weight:0.1f}", 0, (0, 0, 0), COLORS[-1])


    run = True
    while run:

        # GUI management
        for event in pg.event.get():
            # close the GUI
            if event.type == pg.QUIT:
                run = False

            # if button is pressed
            if event.type == pg.KEYDOWN:

                # change beta (temperature)
                if event.key == 273:  # arrow up
                    beta += 1
                    beta_img = beta_txt.render(f"beta: {'0' * (2 - len(str(beta))) + str(beta)}", False, (0, 0, 0),
                                               COLORS[-1])
                elif event.key == 274:  # arrow down
                    beta = max(beta - 1, 0)
                    beta_img = beta_txt.render(f"beta: {'0' * (2 - len(str(beta))) + str(beta)}", False, (0, 0, 0),
                                               COLORS[-1])

                # weight (low-pass filter)
                elif event.key == 119:  # w
                    weight = min(weight + 0.1, 1)
                    weight_img = weight_txt.render(f"weight: {weight:0.1f}", 0, (0, 0, 0), COLORS[-1])
                elif event.key == 115:  # s
                    weight = max(weight - 0.1, 0.1)
                    weight_img = weight_txt.render(f"weight: {weight:0.1f}", 0, (0, 0, 0), COLORS[-1])

                # save image
                elif event.key == 112: # p
                    pg.image.save(display, "triangle_field.png")

        # update field
        t1 = time.time()
        for i in range(times):
            update_field(states, js, beta, loops)
            # apply low-pass filter (IIR)
            avg_states = (1 - weight) * avg_states + weight * states
        time_img = time_txt.render(f"dt: {time.time() - t1:.2f}", False, (0, 0, 0), COLORS[-1])

        # map avg_states (non integer) to spin states {-1, 1}
        display_field = np.where(avg_states > 0, 1, (np.where(avg_states < 0, -1, 0)))
        # check which fields change their state
        result = display_field != old_display_field
        old_display_field = np.copy(display_field)

        # update states
        update_display(cell_grid, result, display_field, display)

        # update information fields
        display.blit(time_img, (30, 30))
        display.blit(beta_img, (30, 50))
        display.blit(weight_img, (30, 70))

        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    blocksize = 20
    n = 20
    width, height = (n + 1) * blocksize, n * blocksize
    angle = 0.66 * np.pi
    mass = 1
    cur_rad = 1
    weight = 1
    loops = int(n / 2 * (n - 1) * 10)
    beta = 6
    bhs = -1
    times = 5
    main(width, height, n, angle, bhs, mass, cur_rad, weight, beta, loops, blocksize, times)
