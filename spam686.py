import logging

import numpy as np

from util import njit_if_numba

NUMBA_AVAILABLE = False
njit = None
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    logging.info("Numba is available...using that for feature extraction")
except:
    logging.info("Numba is not installed...using numpy for feature extraction")
    pass


def spam_extract(img, x=3):
    m3_func = get_m3 if not NUMBA_AVAILABLE else get_m3_numba
    img = np.concatenate((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=1)

    # horizontal left-right
    D = img[:, :-1] - img[:, 1:]
    L, C, R = D[:, 2:], D[:, 1:-1], D[:, :-2]
    mh1 = m3_func(L, C, R, x).flatten("F")

    # horizontal right-left
    D = -D
    L, C, R = D[:, :-2], D[:, 1:-1], D[:, 2:]
    mh2 = m3_func(L, C, R, x).flatten("F")

    #
    D = img[:-1, :] - img[1:, :]
    L, C, R = D[2:, :], D[1:-1, :], D[:-2, :]
    mv1 = m3_func(L, C, R, x).flatten("F")

    D = -D
    L, C, R = D[:-2, :], D[1:-1, :], D[2:, :]
    mv2 = m3_func(L, C, R, x).flatten("F")

    D = img[:-1, :-1] - img[1:, 1:]
    L, C, R = D[2:, 2:], D[1:-1, 1:-1], D[:-2, :-2]
    md1 = m3_func(L, C, R, x).flatten("F")

    D = -D
    L, C, R = D[:-2, :-2], D[1:-1, 1:-1], D[2:, 2:]
    md2 = m3_func(L, C, R, x).flatten("F")

    D = img[1:, :-1] - img[:-1, 1:]
    L, C, R = D[:-2, 2:], D[1:-1, 1:-1], D[2:, :-2]
    mm1 = m3_func(L, C, R, x).flatten("F")

    D = -D
    L, C, R = D[2:, :-2], D[1:-1, 1:-1], D[:-2, 2:]
    mm2 = m3_func(L, C, R, x).flatten("F")

    f1 = (mh1+mh2+mv1+mv2)/4
    f2 = (md1+md2+mm1+mm2)/4
    f = np.concatenate((f1, f2))

    return f


def get_m3(L, C, R, T):
    L = L.flatten()
    L[L < -T] = -T
    L[L > T] = T

    C = C.flatten()
    C[C < -T] = -T
    C[C > T] = T

    R = R.flatten()
    R[R < -T] = -T
    R[R > T] = T

    arr_size = 2*T+1
    M = np.zeros((arr_size, arr_size, arr_size), dtype=np.double)
    for i in range(-T, T+1):
        Li = L==i
        C2 = C[Li]
        R2 = R[Li]
        for j in range(-T, T+1):
            R3 = R2[C2==j]
            for k in range(-T, T+1):
                M[i+T, j+T, k+T] = np.sum(R3==k)

    return M/np.sum(M)


@njit_if_numba(njit, NUMBA_AVAILABLE)
def get_m3_numba(L, C, R, T):
    L = L.flatten()
    L[L < -T] = -T
    L[L > T] = T

    C = C.flatten()
    C[C < -T] = -T
    C[C > T] = T

    R = R.flatten()
    R[R < -T] = -T
    R[R > T] = T

    arr_size = 2*T+1
    M = np.zeros((arr_size, arr_size, arr_size), dtype=np.double)
    
    for i in range(len(L)):
        M[int(L[i])+T, int(C[i])+T, int(R[i])+T] += 1

    return M/np.sum(M)
