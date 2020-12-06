import pygame as pg
import numpy as np
import fastcalc
import time
import random
import math


BLOCKWIDTH = 2
BLOCKHEIGHT = 2
COLORS = {1: (255, 0, 0), -1: (0, 0, 255)}
TIMES = 10
WEIGHT = 0.01

BETA = 5
L = 5

N = 250
M = 1
LOOPS = N * (N+1)
RH = L * math.sqrt(M)


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
    field.fill(np.NaN)
    for y, row in enumerate(field):
        for x in range(y + 2):
            row[x] = d_state
    return field


def prepare(field, mgfield):
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

    for x in range(field.shape[1]):
        if abs((2 * x + 1) / field.shape[1] - 1) < ratio:
            field[-1, x] = -1 * black_hole_spin
        else:
            field[-1, x] = black_hole_spin


def update_display_matrix(avg_field, display_matrix):
    for y in range(avg_field.shape[0]):
        for x in range(y + 2):
            display_matrix[y, x] = 1 if avg_field[y, x] > 0 else -1


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

    field = create_field(N)
    prepare(field, -0.7)
    set_boundary(field, 0.6 * math.pi, -1)

    # access c-extension
    js = fastcalc.get_coupling(N, RH, M, L)

    avgfield = np.copy(field)
    displayfield = np.zeros(avgfield.shape)
    update_display_matrix(avgfield, displayfield)
    old_displayfield = np.copy(displayfield)

    # initiate matrix of rectangles for the presentation of the field
    cells = [[0 for i in range(N + 2)] for j in range(N)]

    for y in range(N):
        x_0 = (N - y) / 2
        for x in range(y + 2):
            # create a surface for each rectangle (this will only be updated if necessary)
            area = pg.Surface((BLOCKWIDTH, BLOCKHEIGHT)).convert()
            area.fill(COLORS[displayfield[y, x]])
            cells[y][x] = (area, area.get_rect())
            cells[y][x][1].center = ((x + x_0) * BLOCKWIDTH, y * BLOCKHEIGHT)
            display.blit(cells[y][x][0], cells[y][x][1])
        pg.display.flip()

    run = True
    while run:
        # check for close order
        for event in pg.event.get():
            if event.type == pg.QUIT:
                # break is not possible because of pygame (may not work)
                run = False

        t1 = time.time()
        for i in range(TIMES):
            fastcalc.update(field, js, N, LOOPS, BETA)
            avgfield = (1 - WEIGHT) * avgfield + WEIGHT * field
        print(time.time() - t1)

        update_display_matrix(avgfield, displayfield)
        result = displayfield != old_displayfield

        for pos in np.argwhere(result):
            cells[pos[0]][pos[1]][0].fill(COLORS[displayfield[pos[0], pos[1]]])
            display.blit(cells[pos[0]][pos[1]][0], cells[pos[0]][pos[1]][1])
        old_displayfield = np.copy(displayfield)

        pg.display.flip()


if __name__ == "__main__":
    main(BLOCKWIDTH * N + 2, BLOCKHEIGHT * N)