import numpy as np


def trace(d, A):
    tr = 0
    for j in range(0, d):
        tr += A[j, j]
    return tr


def pTraceL(dl, dr, rhoLR):
    # Returns the left partial trace over the 'left' subsystem of rhoLR
    rhoB = np.zeros((dr, dr), dtype=complex)
    for j in range(0, dr):
        for k in range(j, dr):
            for l in range(0, dl):
                rhoB[j][k] += rhoLR[l*dr+j][l*dr+k]
            if j != k:
                rhoB[k][j] = np.conj(rhoB[j][k])
    return rhoB


def pTraceR(dl, dr, rhoLR):
    # Returns the right partial trace over the 'right' subsystem of rhoLR
    rhoA = np.zeros((dl, dl), dtype=complex)
    for j in range(0, dl):
        for k in range(j, dl):
            for l in range(0, dr):
                rhoA[j][k] += rhoLR[j*dr+l][k*dr+l]
        if j != k:
            rhoA[k][j] = np.conj(rhoA[j][k])
    return rhoA


def pTraceM(dl, dm, dr, rhoLMR):
    # Returns the partial trace over the middle subsystem of rhoLMR
    dlr = dl*dr
    rhoLR = np.zeros((dlr, dlr), dtype=complex)
    for j in range(0, dl):
        for l in range(0, dr):
            cj = j*dr + l
            ccj = j*dm*dr + l
            for m in range(0, dl):
                for o in range(0, dr):
                    ck = m*dr + o
                    cck = m*dm*dr + o
                    for k in range(0, dm):
                        rhoLR[cj][ck] += rhoLMR[ccj+k*dr][cck+k*dr]
    return rhoLR

def pTraceL_num(dl, dr, rhoLR):
    # Returns the left partial trace over the 'left' subsystem of rhoLR
    rhoR = np.zeros((dr, dr), dtype=complex)
    for j in range(0, dr):
        for k in range(j, dr):
            for l in range(0, dl):
                rhoR[j,k] += rhoLR[l*dr+j,l*dr+k]
            if j != k:
                rhoR[k,j] = np.conj(rhoR[j,k])
    return rhoR

def pTraceR_num(dl, dr, rhoLR):
    # Returns the right partial trace over the 'right' subsystem of rhoLR
    rhoL = np.zeros((dl, dl), dtype=complex)
    for j in range(0, dl):
        for k in range(j, dl):
            for l in range(0, dr):
                rhoL[j,k] += rhoLR[j*dr+l,k*dr+l]
        if j != k:
            rhoL[k,j] = np.conj(rhoL[j,k])
    return rhoL
