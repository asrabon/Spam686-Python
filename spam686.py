import numpy as np


def spam_extract(img, x=3):
    img = np.concatenate((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=1)

    # horizontal left-right
    D = img[:, :-1] - img[:, 1:]
    L, C, R = D[:, 2:], D[:, 1:-1], D[:, :-2]
    mh1 = get_m3(L, C, R, x)

    # horizontal right-left
    D = -D
    L, C, R = D[:, :-2], D[:, 1:-1], D[:, 2:]
    mh2 = get_m3(L, C, R, x)

    #
    D = img[:-1, :] - img[1:, :]
    L, C, R = D[2:, :], D[1:-1, :], D[:-2, :]
    mv1 = get_m3(L, C, R, x)

    D = -D
    L, C, R = D[:-2, :], D[1:-1, :], D[2:, :]
    mv2 = get_m3(L, C, R, x)

    D = img[:-1, :-1] - img[1:, 1:]
    L, C, R = D[2:, 2:], D[1:-1, 1:-1], D[:-2, :-2]
    md1 = get_m3(L, C, R, x)

    D = -D
    L, C, R = D[:-2, :-2], D[1:-1, 1:-1], D[2:, 2:]
    md2 = get_m3(L, C, R, x)

    D = img[1:, :-1] - img[:-1, 1:]
    L, C, R = D[:-2, 2:], D[1:-1, 1:-1], D[2:, :-2]
    mm1 = get_m3(L, C, R, x)

    D = -D
    L, C, R = D[2:, :-2], D[1:-1, 1:-1], D[:-2, 2:]
    mm2 = get_m3(L, C, R, x)

    f1 = (mh1+mh2+mv1+mv2)/4
    f2 = (md1+md2+mm1+mm2)/4
    f = np.concatenate((f1, f2))

    return f


def get_m3(L, C, R, T):
    L = L.flatten()
    #np.where(L < -T, L, -T)
    #np.where(L > T, L, T)
    L[L < -T] = -T
    L[L > T] = T

    C = C.flatten()
    #np.where(C<-T, C, -T)
    #np.where(C > T, C, T)
    C[C < -T] = -T
    C[C > T] = T

    R = R.flatten()
    #np.where(R < -T, R, -T)
    #np.where(R > T, R, T)
    R[R < -T] = -T
    R[R > T] = T

    arr_size = 2*T+1
    M = np.zeros((arr_size, arr_size, arr_size), dtype=np.double)
    for i in range(-T, T+1):
        C2 = C[L==i]
        R2 = R[L==i]
        for j in range(-T, T+1):
            R3 = R2[C2==j]
            for k in range(-T, T+1):
                M[i+T, j+T, k+T] = np.sum(R3==k)

    return M.flatten("F")/np.sum(M)
