from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify
#from sympy.physics.quantum import TensorProduct, Dagger
import numpy as np
from numpy import linspace
#import matplotlib.pyplot as plt
#import math
#from theoric.tools import *
#import torch
from torch import tensor

def get_list_p_noMarkov(list_p, type):
        lamb = 0.01
        gamma_0 = 2
        list_p_noMarkov = []
        if type == 'Bellomo':
            def non_markov_list_p(lamb,gamma_0,t):
                d = sqrt(2*gamma_0*lamb-lamb**2)
                result = exp(-lamb*t)*(cos(d*t/2)+(lamb/d)*sin(d*t/2))**2
                return result
        if type == 'Ana':
            def non_markov_list_p(lamb,gamma_0,t):
                result = 1-exp(-lamb*t)*(cos(t/2)+(lamb)*sin(t/2))
                return result
        for p in list_p:
            list_p_noMarkov.append(non_markov_list_p(lamb,gamma_0,p))
        return list_p_noMarkov

class QuantumChannels(object):
    def __init__(self):
        theta = Symbol('theta',real=True)
        phi = Symbol('phi',real=True)
        gamma = Symbol('gamma',real=True, positive=True)
        p = Symbol('p',real=True, positive=True)

    def get_target_op(state):
        M_numpy = np.array(state.tolist(), dtype=np.complex64)
        rho = simplify(M_numpy)
        state2 = np.zeros(np.shape(rho)[1],dtype=complex)
        aux = 0
        for i in rho[0]:
            state2[aux] = i
            aux += 1
        target_op = np.outer(state2.conj(), state2)
        target_op = tensor(target_op)
        return target_op
    
    def rho_AB_ad(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        0]])
        return state
    
    def rho_AB_bpf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*(cos(theta/2))),
                        (-1j*sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        (1j*sqrt(p)*cos(theta/2))]])
        return state

    def rho_AB_bf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*cos(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        sqrt(p)*cos(theta/2)]])
        return state

    def rho_AB_pf(theta, phi, p):
        state = Matrix([[(sqrt(1-p)*cos(theta/2)),
                        +(sqrt(p)*cos(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        -sqrt(p)*exp(1j*phi)*sin(theta/2)]])
        return state

    def rho_AB_pd(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                         0,
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2))]])
        return state
    
    @staticmethod
    def rho_AB_l(theta, phi, p):
        state = Matrix([[sqrt(1/2)*(cos(p/2)*cos(theta/2) + sin(p/2)*exp(1j*phi)*sin(theta/2)),
                         sqrt(1/2)*(cos(p/2)*cos(theta/2) - sin(p/2)*exp(1j*phi)*sin(theta/2)),
                         sqrt(1/2)*(cos(p/2)*exp(1j*phi)*sin(theta/2) - sin(p/2)*cos(theta/2)),
                         sqrt(1/2)*(cos(p/2)*exp(1j*phi)*sin(theta/2) + sin(p/2)*cos(theta/2))
                         ]])
        return state

    @staticmethod
    def rho_AB_d(theta, phi, p):
        state = Matrix([[(sqrt(1-3*p/4)*cos(theta/2)),
                        (sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
                        (-1j*sqrt(p/4)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p/4)*cos(theta/2)),
                        (sqrt(1-3*p/4)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(p/4)*cos(theta/2)),
                        (1j*sqrt(p/4)*cos(theta/2)),
                        (-sqrt(p/4)*exp(1j*phi)*sin(theta/2))
                        ]])
        return state
    @staticmethod
    def rho_AB_adg(theta, phi, p):#, gamma):
        N = 0.5
        state = Matrix([[sqrt(1-N)*cos(theta/2),# |000\rangle \\
                         sqrt(p*(1-N))*exp(1j*phi)*sin(theta/2),# |001\rangle\\
                         sqrt(N*(1-p))*cos(theta/2),# |010\rangle\\
                         0,# |011\rangle,
                         sqrt((1-N)*(1-p))*exp(1j*phi)*sin(theta/2),# |100\rangle\\
                         0,
                         sqrt(N)*exp(1j*phi)*sin(theta/2),# |110\rangle\\
                         sqrt(p*N)*cos(theta/2)# |111\rangle
                        ]])
        return state
    
    @staticmethod
    def rho_AB_hw_ref(theta, phi, p):#, gamma):
        N = 0.5
        state = Matrix([[sqrt(p/3), # |0000\rangle 
                         sqrt((1-p)/6), # |0001\rangle 
                         sqrt((1-p)/6), # |0010\rangle 
                         0, # |0011\rangle 
                         sqrt(p/3), # |0100\rangle 
                         sqrt(((1-p)*((1j*sqrt(3)-1)**2))/24), # |0101\rangle 
                         sqrt(((1-p)*((-1j*sqrt(3)-1)**2))/24), # |0110\rangle 
                         0, # |0111\rangle 
                         sqrt(p/3), # |1000\rangle 
                         sqrt(((1-p)*((-1j*sqrt(3)-1)**2))/24), # |1001\rangle 
                         sqrt(((1-p)*((1j*sqrt(3)-1)**2))/24), # |1010\rangle 
                         0, # |1011\rangle 
                         0, # |1100\rangle 
                         0, # |1101\rangle 
                         0, # |1110\rangle 
                         0, # |1111\rangle
                        ]])
        return state


    @staticmethod
    def rho_AB_hw(theta, phi, p):#, gamma):
        N = 0.5
        p0 = p
        p1 = (1-p0)/2
        p2 = (1-p0)/2
        state = Matrix([[sqrt(p0/3), # |0000\rangle 
                         sqrt(p1/3), # |0001\rangle 
                         sqrt(p2/3), # |0010\rangle 
                         0, # |0011\rangle 
                         sqrt(p0/3), # |0100\rangle 
                         sqrt(p1/3)*exp(2*1j*pi/3), # |0101\rangle 
                         sqrt(p2/3)*exp(4*1j*pi/3), # |0110\rangle 
                         0, # |0111\rangle 
                         sqrt(p0/3), # |1000\rangle 
                         sqrt(p1/3)*exp(4*1j*pi/3), # |1001\rangle 
                         sqrt(p2/3)*exp(8*1j*pi/3), # |1010\rangle 
                         0, # |1011\rangle 
                         0, # |1100\rangle 
                         0, # |1101\rangle 
                         0, # |1110\rangle 
                         0, # |1111\rangle
                        ]])
        return state

    def show_eq(rho, theta=None, phi=None, gamma=None, p=None):
        if theta == None or phi == None or gamma == None or p == None:
            theta = Symbol('theta',real=True)
            phi = Symbol('phi',real=True)
            gamma = Symbol('gamma',real=True, positive=True)
            p = Symbol('p',real=True, positive=True)
        print(rho(theta, phi, p))

    def show_eq_with_gamma(rho, theta=None, phi=None, gamma=None, p=None):
        if theta == None or phi == None or gamma == None or p == None:
            theta = Symbol('theta',real=True)
            phi = Symbol('phi',real=True)
            gamma = Symbol('gamma',real=True, positive=True)
            p = Symbol('p',real=True, positive=True)
        print(rho(theta, phi, p, gamma))


def main():
    from sympy import Symbol,simplify,print_latex
    import sys
    sys.path.append('runtime-qiskit')
    sys.path.append('src')
    from theoric_channels import TheoricMaps
    a = TheoricMaps()
    theta = Symbol('theta',real=True)
    phi = Symbol('phi',real=True)
    gamma = Symbol('gamma',real=True, positive=True)
    p = Symbol('p',real=True, positive=True)
    from kraus_maps import QuantumChannels
    a = QuantumChannels()
    print_latex(simplify(a.rho_AB_hw(theta,phi,p)))

if __name__ == "__main__":
    main()
#a=10
#QCH = QuantumChannels()
#a = QCH.rho_AB_bpf
#print(a(0,0,0))
#print(QCH.get_target_op(QCH.rho_AB_pd(pi/2,0,0)))
