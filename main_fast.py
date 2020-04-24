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
WEIGHT = 1 / TIMES

BETA = 5
L = 5
N = 250
M = 1
LOOPS = N * (N-2)
print(f"Elements: {LOOPS}")

RH = L / math.sqrt(M)
PREF_JH = np.pi * L / (2 * N)
PREF_JV = 2 * math.pi * RH / N
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
    js = fastcalc.get_coupling(N, RH, M, L)

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
            area.fill(COLORS[displayfield[y, x]])
            temp.append((area, area.get_rect()))
            temp[-1][1].center = (x * BLOCKWIDTH, y * BLOCKHEIGHT)
            display.blit(temp[-1][0], temp[-1][1])
        cells.append(temp)
    pg.display.flip()


    run = True
    while run:
        # check for close order
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        t1 = time.time()
        for i in range(TIMES):
            fastcalc.update(field, js, N, LOOPS, BETA)
            avgfield += WEIGHT * field
        print(time.time() - t1)

        displayfield = np.where(avgfield > 0, 1, -1)
        result = displayfield == olddisplayfield
        for pos in np.argwhere(result):
            cells[pos[0]][pos[1]][0].fill(COLORS[displayfield[pos[0], pos[1]]])
            display.blit(cells[pos[0]][pos[1]][0], cells[pos[0]][pos[1]][1])
        olddisplayfield = np.copy(displayfield)

        pg.display.flip()


if __name__ == "__main__":
    main(BLOCKWIDTH * N, BLOCKHEIGHT * N)
