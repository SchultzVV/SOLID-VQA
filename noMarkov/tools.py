'''   ENTROPY.PY                 '''
import math
import scipy.linalg.lapack as lapak
import sys
sys.path.append('..')
import src.pTrace

def purity(rho):
    d = rho.shape[0]
    purity = 0.0
    j = -1
    while (j < d-1):
        j = j + 1
        k = -1
        while (k < d-1):
            k = k + 1
            purity += (rho[j][k].real)**2 + (rho[j][k].imag)**2
    return purity

def linear_entropy(rho):
    return 1-purity(rho)

def shannon(pv):
    d = pv.shape[0]
    SE = 0.0
    j = -1
    while (j < d-1):
        j = j + 1
        if pv[j] > 1.e-15 and pv[j] < (1.0-1.e-15):
            SE -= pv[j]*math.log(pv[j], 2)
    return SE

def von_neumann(rho):
    d = rho.shape[0]
    b = lapak.zheevd(rho)
    VnE = shannon(b[0])
    return VnE

def mutual_info(rho, dl, dr):
    rhor = pTrace.pTraceL(dl, dr, rho)
    rhol = pTrace.pTraceR(dl, dr, rho)
    return von_neumann(rhol) + von_neumann(rhor) - von_neumann(rho)

'''  ENTANGLEMENT.PY   '''
import numpy as np
from numpy import linalg as LA
from math import sqrt


def concurrence(rho):
    print(rho)
    ev = np.zeros(4, dtype='float')
    R = np.zeros((4, 4), dtype=complex)
    R = np.dot(rho, np.kron(Pauli(2), Pauli(2)))
    R = np.dot(R, np.conj(rho))
    R = np.dot(R, np.kron(Pauli(2), Pauli(2)))
    ev = LA.eigvalsh(R)
    evm = max(abs(ev[0]), abs(ev[1]), abs(ev[2]), abs(ev[3]))
    C = 2.0*sqrt(abs(evm)) - sqrt(abs(ev[0]))
    C = C - sqrt(abs(ev[1])) - sqrt(abs(ev[2])) - sqrt(abs(ev[3]))
    if C < 0.0:
        C = 0.0
    return C


def EoF(rho):
    pv = np.zeros(2)
    Ec = concurrence(rho)
    pv[0] = (1.0 + np.sqrt(1.0 - Ec**2.0))/2.0
    pv[1] = 1.0 - pv[0]
    EF = shannon(2, pv)
    return EF


def negativity(d, rhoTp):
    from distances import normTr
    En = 0.5*(normTr(d, rhoTp) - 1.0)
    return En


def log_negativity(d, rhoTp):
    En = negativity(d, rhoTp)
    Eln = np.log(2.0*En+1.0, 2)
    return Eln


def chsh(rho):  # arXiv:1510.08030
    import gell_mann as gm
#    cm = np.zeros(3, 3)
    cm = gm.corr_mat(2, 2, rho)
    W = np.zeros(3)
    # W = LA.eigvalsh(cm)
    u, W, vh = LA.svd(cm, full_matrices=True)
    no = np.sqrt(2)-1
    nl = (sqrt(W[0]**2+W[1]**2+W[2]**2-min(W[0], W[1], W[2])**2)-1)/no
    return max(0, nl)


def steering(rho):  # arXiv:1510.08030
    import gell_mann as gm
#    cm = np.zeros(3,3)
    cm = gm.corr_mat(2, 2, rho)
    W = np.zeros(3)
    # W = LA.eigvalsh(cm)
    u, W, vh = LA.svd(cm, full_matrices=True)
    return max(0, (sqrt((W[0]**2)+(W[1]**2)+(W[2]**2))-1)/(sqrt(3)-1))
