from letalker import core
import numpy as np
# import numba as nb


# def test_get_param():

#     A = np.zeros((100, 2, 3))
#     Ai = np.empty((2, 3))
#     get_param2(Ai.ctypes, A.ctypes, 40, *A.shape)

#     assert np.array_equal(A[40], Ai)


@nb.jit(nopython=True)
def get_param2_njit(A, i):

    Ai = np.empty((2, 3))
    core.get_param2(Ai.ctypes, A.ctypes, i, *A.shape)

    return Ai


def test_get_param_njit():
    A = np.zeros((100, 2, 3))

    Ai = get_param2_njit(A, 40)
    assert np.array_equal(A[40], Ai)
