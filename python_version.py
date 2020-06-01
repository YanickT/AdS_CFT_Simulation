import pygame as pg
import numpy as np
import random
import time
import math
from numba import njit


BLOCKWIDTH = 5
BLOCKHEIGHT = 5
COLORS = {1: (255, 0, 0), -1: (0, 0, 255), 0: (255, 255, 255)}

BETA = 10
TIMES = 10
WEIGHT = 1 #0.0005

L = 1
N = 120
M = 1
LOOPTIMES = 10
LOOPS = N * (N - 2) * LOOPTIMES
PHI = 0.6 * math.pi

RH = L * math.sqrt(M)
PREF_JH = np.pi * L / (2 * N)
PREF_JV = 2 * math.pi * RH / N
PI_2N = math.pi / (N * 2)

C1 = PHI / (2 * np.sin(PHI/2))
C2 = C1 * np.cos(PHI/2) - math.pi / 2
i_phi = lambda phi: int((phi * N / math.pi - 1) / 2)
j_eta = lambda eta: int((4 * eta * N / math.pi - 1) / 2)
s_i = lambda i: math.pi * (2 * i + 1) / N
eta_s = lambda s: 2 * C1 * np.cos(s) - C2
phi_s = lambda s: C1 * np.sin(s)
phi_0 = PHI/(2 * np.pi)
a = round(0.5 * ((1 - phi_0) * N - 1))
b = round(0.5 * ((phi_0 + 1) * N - 1))
x_0 = (b + a) // 2
POINTS = [(i_phi(phi_s(s_i(i))) + x_0, 40 + N + j_eta(eta_s(s_i(i)))) for i in range(a, b)]
xs, ys = tuple(zip(*POINTS))
print(max(xs), min(xs))
print(max(ys), min(ys))


def prepare(field, mgfield):
    """
    Set every state to a random state determined by mgfield.
    :param field: 2d array: state field of the Ising model
    :param mgfield: float: default magnetic field
    :return: void
    """
    field.fill(-1)
    for y in range(N):
        for x in range(N):
            if random.random() < (mgfield + 1) / 2:
                field[y, x] = 1


def get_couplings():
    """
    Function calculates the coupling constants for each position for its neighbours.
    :return: np.array of shape (N, N, 3). array[y, x, (horizontal | up | down)]
    """
    js = np.zeros((N, N, 3))
    for y in range(N):
        j_horizontal = PREF_JH / math.cos(PI_2N * (y + 0.5))
        for x in range(N):
            js[y, x, 0] = j_horizontal
            js[y, x, 1] = PREF_JV / math.cos(PI_2N * y + PI_2N)
            js[y, x, 2] = PREF_JV / math.cos(PI_2N * y)

            # normalize to max_field
            js[y, x, :] /= np.sum(js[y, x, :]) + js[y, x, 0]

    return js


def set_boundary(field, angle, black_hole_spin):
    """
    Force the given boundary conditions to the given field.
    :param field: np.array of shape (N, N) = state field of the Ising model
    :param angle: in [0, 2*pi] = Angle of the entangelment
    :param black_hole_spin: in {-1, 1} = Spin orientation of the centred balckhole
    :return: void
    """
    ratio = angle / (2 * np.pi)
    for i in range(N):
        field[0][i] = black_hole_spin
        if abs((2 * i + 1) / N - 1) < ratio:
            field[-1][i] = -1 * black_hole_spin
        else:
            field[-1][i] = black_hole_spin


@njit()
def loop2(field, js, loops):
    for i in range(loops):
        # select random position
        x, y = int(random.random() * N), int(random.random() * (N - 2) + 1)
        h = js[y, x, 0] * field[y, (x + 1) % N] + js[y, x, 0] * field[y, (x + N - 1) % N] + \
            js[y, x, 1] * field[y + 1, x] + js[y, x, 2] * field[y - 1, x]
        field[y, x] = 1 if random.random() < 1 / (1 + np.exp(-2 * BETA * h)) else -1


def draw_expection(display_field):
    for x, y in POINTS:
        display_field[y, x] = 0


def main(width, height):
    """
    Start function of the simulation.
    :param width: width of the widget
    :param height: height of the widget
    :return: void
    """
    # init pygame
    pg.init()
    pg.display.set_caption('Ryu-Takayanagi Ising Simulator using Pygame')
    display = pg.display.set_mode((width, height))

    # create field, get it ready and calculate coupling constants
    field = np.zeros((N, N), dtype=np.int32)
    prepare(field, -0.7)
    set_boundary(field, PHI, -1)
    js = get_couplings()

    avgfield = np.copy(field)
    displayfield = np.where(avgfield > 0, 1, -1)
    olddisplayfield = np.copy(displayfield)


    # initiate matrix of rectangles for the presentation of the field
    cells = []
    for y in range(N):
        temp = []
        for x in range(N):
            # create a surface for each rectangle (this will only be updated if necessary)
            area = pg.Surface((BLOCKWIDTH, BLOCKHEIGHT)).convert()
            area.fill(COLORS[field[y, x]])
            temp.append((area, area.get_rect()))
            temp[-1][1].center = (x * BLOCKWIDTH, y * BLOCKHEIGHT)
            display.blit(temp[-1][0], temp[-1][1])
        cells.append(temp)

    run = True
    while run:
        # check for close order
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        t1 = time.time()
        for i in range(TIMES):
            loop2(field, js, LOOPS)
            avgfield = WEIGHT * field + (1 - WEIGHT) * avgfield
        print(time.time() - t1)

        displayfield = np.where(avgfield > 0, 1, -1)
        draw_expection(displayfield)
        result = displayfield == olddisplayfield
        olddisplayfield = np.copy(displayfield)

        for pos in np.argwhere(result):
            cells[pos[0]][pos[1]][0].fill(COLORS[displayfield[pos[0], pos[1]]])
            display.blit(cells[pos[0]][pos[1]][0], cells[pos[0]][pos[1]][1])

        pg.display.update()


if __name__ == "__main__":
    main(BLOCKWIDTH * N, BLOCKHEIGHT * N)