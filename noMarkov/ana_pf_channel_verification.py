from src.tools import *
import sys as s
from sympy import *
from scipy.linalg import sqrtm
import numpy as np
init_printing(use_unicode=True)
from matplotlib import pyplot as plt
#%matplotlib inline
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import TensorProduct
from scipy.linalg import eigvals

import scipy.interpolate
import platform

def werner_state(c1, c2, c3):
    # c = [-0.8,-0.8,-0.8]
    c = [c1, c2, c3]
    index = 0
    rho = np.zeros((4,4),dtype=complex)
    for i in range(len(rho)):
        rho[i,i] = 1
    for i in c:
        index += 1
        rho += TensorProduct(i*Pauli(index),Pauli(index))
    #print(np.array(rho,dtype=complex))
    return rho
werner = werner_state(-0.8, -0.8, -0.8)
#stical modeling 
Mais=(cb(2,0)+cb(2,1))/sqrt(2)
Menos=(cb(2,0)-cb(2,1))/sqrt(2)
#werner

'''phase flip'''
def K_0(J):
    return sqrt(1-J/2)*Pauli(0)

def K_1(J):
    return sqrt(J/2)*Pauli(3)

def TP(a,b):
    return TensorProduct(a,b)

def proj(psi):
    z = Dagger(psi)
    return psi*z

# função pra obter o estado evoluído
def RHO_t_NM(state,J):
    tp1 = TP(K_0(J),K_1(J))
    tp2 = TP(K_1(J),K_0(J))
    return tp1*proj(state)*tp1.T + tp2*proj(state)*tp2.T

def calculate_entanglement(rho):
    # Compute the spin-flipped counterpart
    sigma_y = Matrix([[0, -1j], [1j, 0]])
    rho_tilde = np.kron(sigma_y, sigma_y) @ np.conj(rho.T) @ np.kron(sigma_y, sigma_y)

    # Convert rho_tilde to a NumPy array
    rho_tilde_numpy = np.array(rho_tilde.tolist(), dtype=np.complex128)

    # Compute the eigenvalues of the modified matrix
    eigenvalues = eigvals(rho_tilde_numpy)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

    # Calculate the entanglement measure
    entanglement = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

    return entanglement


def get_list_p_noMarkov(list_p):
    lamb = 0.01
    list_p_noMarkov = []
    def non_markov_p(lamb,t):
        result = 1-(exp(-lamb*t)*(cos(t/2)+(lamb)*sin(t/2)))
        return result
    for p in list_p:
        list_p_noMarkov.append(non_markov_p(lamb,p))
    return list_p_noMarkov

T = np.linspace(0.01,200,1000)
t_A = get_list_p_noMarkov(T)
print(T)
print(t_A)

state = werner_state(-0.8,-0.8,-0.8)
print(werner_state(-0.8,-0.8,-0.8))
p=0
state_t = werner_state_t(-0.8,-0.8,-0.8,p)
print('werner_state',state)
print('werner_state state =',type(state))
print('----------------------------------')
print('werner_state',state_t)
print('werner_state state =',type(state_t))
#print(RHO_t_NM(state, 14))
#print(type(RHO_t_NM(state, 14)))
print(calculate_entanglement(state_t))

y3 = [calculate_entanglement(werner_state_t(-0.8,-0.8,-0.8, i)) for i in t_A]

plt.plot(T,y3,label='entanglement - Ana')

plt.xscale('log')
plt.xlabel('log(t)')

# plt.xlim(0.01, 200)
plt.grid(True)
plt.legend()
plt.show()
s.exit()
#print('RHO_t_NM(state, 14) =',type(RHO_t_NM(state, 14)))

# y1 = [coh_l1(RHO_t_NM(state, i)) for i in t_A]
y1 = [calculate_entanglement(RHO_t_NM(state, i)) for i in t_A]
y2 = [coh_l1(RHO_t_NM(state, i)) for i in t_A]


#T = [ np.log(i) for i in t_A]
plt.plot(T,y1,label='entanglement - Ana')
plt.plot(T,y2,label='coh_l1 - Ana')
plt.plot(T,y3,label='concurrence - Ana')

plt.xscale('log')
plt.xlabel('log(t)')

# plt.xlim(0.01, 200)
plt.grid(True)
plt.legend()
plt.show()