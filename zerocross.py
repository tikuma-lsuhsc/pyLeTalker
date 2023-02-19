import numpy as np


def zerocross(x, Fs):
    #
    Dt = 1 / Fs
    t = np.arange(len(x) - 1) / Fs

    tme = t[(x[:-1] < 0) & (x[1:] >= 0)] + Dt / 2

    if len(tme):
        fo = 1 / np.diff(tme)
        fo = np.array([fo[0], *fo, fo[-1]])
        tme = np.array([0.0, *tme])
    else:
        fo = 0

    return fo, tme
