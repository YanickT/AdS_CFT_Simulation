import pygame as pg
import numpy as np
import random
import fastrand
import time
import math


fastrand.pcg32_seed(time.time())


BLOCKWIDTH = 5
BLOCKHEIGHT = 5
COLORS = {1: (255, 0, 0), -1: (0, 0, 255)}


T = 50
BETA = 5

L = 5
N = 120
M = 1
LOOPS = 10000


PREF_JH = np.pi * L / (2 * N)
PREF_JV = 2 * math.pi * T / N
PI_2N = math.pi / (N * 2)


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

    print(js)
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
    field = np.zeros((N, N))
    prepare(field, -0.7)
    set_boundary(field, 0.6 * math.pi, -1)
    js = get_couplings()

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

        for i in range(LOOPS):
            # select random position
            x, y = fastrand.pcg32bounded(N), fastrand.pcg32bounded(N - 2) + 1
            h = js[y, x, 0] * field[y, (x + 1) % N] + js[y, x, 0] * field[y, (x + N - 1) % N] + \
                js[y, x, 1] * field[y + 1, x] + js[y, x, 2] * field[y - 1, x]
            if random.random() < 1 / (1 + np.exp(-2 * BETA * h)):
                if field[y, x] != 1:
                    field[y, x] = 1
                    cells[y][x][0].fill(COLORS[1])
                    display.blit(cells[y][x][0], cells[y][x][1])
            elif field[y, x] != -1:
                field[y, x] = - 1
                cells[y][x][0].fill(COLORS[-1])
                display.blit(cells[y][x][0], cells[y][x][1])

        pg.display.update()


if __name__ == "__main__":
    main(BLOCKWIDTH * N, BLOCKHEIGHT * N)
