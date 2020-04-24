import time
import fastcalc
import numpy as np
import math, random

T = 50
BETA = 5

L = 5
N = 120
M = 1
LOOPS = 50000


PREF_JH = np.pi * L / (2 * N)
PREF_JV = 2 * math.pi * T / N
PI_2N = math.pi / (N * 2)

def get_couplings():
    """
    Function calculates the coupling constants for each position for its neighbours.
    :return: np.array of shape (N, N, 3). array[y, x, (horizontal | up | down)]
    """
    js = np.zeros((N, N, 3))
    for y in range(N):
        j_horizontal = PREF_JH / math.cos(PI_2N * (y + 0.5))
        j_ver_up = PREF_JV / math.cos(PI_2N * y + PI_2N)
        j_ver_down = PREF_JV / math.cos(PI_2N * y)
        for x in range(N):
            js[y, x, 0] = j_horizontal
            js[y, x, 1] = j_ver_up
            js[y, x, 2] = j_ver_down

            # normalize to max_field
            js[y, x, :] /= np.sum(js[y, x, :]) + js[y, x, 0]
    return js


r1 = get_couplings().tolist()
r2 = fastcalc.get_coupling(N, T, M, L).tolist()

for y in range(N):
    for x in range(N):
        print(r1[y][x])
        print(r2[y][x])
        print("")
    input()



