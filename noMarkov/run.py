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
#print(werner)
#s.exit()

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

print(RHO_t_NM(werner,1))
s.exit()
def calculate_entanglement_old(rho):
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


def is_numeric_matrix(matrix):
    """Check if the given matrix is numeric."""
    if not isinstance(matrix, (np.ndarray, np.matrix)):
        return False
    if not np.issubdtype(matrix.dtype, np.number):
        return False
    return True

def is_hermitian(matrix):
    """Check if the given matrix is Hermitian (self-adjoint)."""
    return np.allclose(matrix, matrix.conj().T)

def has_non_numeric_values(matrix):
    """Check if the given matrix has non-numeric values."""
    return np.any(np.isnan(matrix)) or np.any(np.isinf(matrix))

def calculate_entanglement2(rho):
    """Calculates the von Neumann entropy of a quantum state using the alternative formula.
    
    Parameters:
    rho (array-like or Matrix): Density operator of the quantum state.
    
    Returns:
    float: Von Neumann entropy calculated using the alternative formula.
    """
    rho_array = np.array(rho)  # Convert to NumPy array
    
    #if not is_numeric_matrix(rho_array) or has_non_numeric_values(rho_array):
    #    raise ValueError("The input matrix must be numeric and free of non-numeric values.")
    
    #if not is_hermitian(rho_array):
    #    raise ValueError("The input matrix must be Hermitian (self-adjoint).")
    
    # Construct the spin-flipped counterpart of rho
    sigma_y = np.array([[0, -1j], [1j, 0]])
    rho_tilde = np.kron(sigma_y, sigma_y) @ rho_array.conj().T @ np.kron(sigma_y, sigma_y)
    
    # Calculate the eigenvalues of rho_tilde
    eigenvalues = np.linalg.eigvalsh(rho_tilde)
    
    # Sort the eigenvalues in descending order
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    # Calculate the alternative von Neumann entropy formula
    entropy = max(0, np.sqrt(eigenvalues_sorted[0]) - np.sqrt(eigenvalues_sorted[1]) -
                   np.sqrt(eigenvalues_sorted[2]) - np.sqrt(eigenvalues_sorted[3]))
    
    return entropy


# Função para calcular o emaranhamento
def calculate_entanglement(rho):
    rho_sqrt = rho.applyfunc(sympify)  # Convert all matrix elements to sympy expressions
    eigenvalues = rho_sqrt.eigenvals()  # Calculate eigenvalues using SymPy's eigenvals method

    eigenvalues_real = [val for val in eigenvalues if val.is_real]
    eigenvalues_complex = [val for val in eigenvalues if not val.is_real]
    
    def custom_max(iterable):
        max_val = None
        for val in iterable:
            if max_val is None or val > max_val:
                max_val = val
        return max_val
    
    max_real = custom_max(eigenvalues_real)
    max_complex = custom_max(eigenvalues_complex)

    entanglement = max(0, max_real - sum([sqrt(val) for val in eigenvalues_complex]))
    return entanglement

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

T = np.linspace(0.01,1000,200)
t_A = get_list_p_noMarkov(T, 'Ana')
t_B = get_list_p_noMarkov(T, 'Bellomo')
#t_A = T

state = werner_state(-0.8,-0.8,-0.8)
state = werner_state(-0.8,-0.8,-0.8)
print('werner_state',state)
print('werner_state state =',type(state))
print(RHO_t_NM(state, 14))
print(type(RHO_t_NM(state, 14)))
print(calculate_entanglement(RHO_t_NM(state, 14)))
print('RHO_t_NM(state, 14) =',type(RHO_t_NM(state, 14)))
# s.exit()

# y1 = [coh_l1(RHO_t_NM(state, i)) for i in t_A]
y1 = [calculate_entanglement(RHO_t_NM(state, i)) for i in t_A]
y2 = [coh_l1(RHO_t_NM(state, i)) for i in t_A]
y3 = [concurrence(RHO_t_NM(state, i)) for i in t_A]

y4 = [calculate_entanglement(RHO_t_NM(state, i)) for i in t_B]
y5 = [coh_l1(RHO_t_NM(state, i)) for i in t_B]
y6 = [concurrence(RHO_t_NM(state, i)) for i in t_B]

#T = [ np.log(i) for i in t_A]
plt.plot(T,y1,label='entanglement - Ana')
plt.plot(T,y2,label='coh_l1 - Ana')
plt.plot(T,y3,label='concurrence - Ana')

# plt.plot(T,y4,label='entanglement - Bellomo')
# plt.plot(T,y5,label='coh_l1 - Bellomo')
# plt.plot(T,y6,label='concurrence - Bellomo')

plt.xscale('log')
plt.xlabel('log(t)')

# plt.xlim(0.01, 200)
plt.grid(True)
plt.legend()
plt.show()