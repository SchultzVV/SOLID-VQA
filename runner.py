from src.simulation_with_save import Simulate
from kraus_maps import QuantumChannels as QCH
import numpy as np
from numpy import pi

''' Com a ref de Ana Costa'''
#---------AD------------------
n_qubits = 2
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_ad
S = Simulate('ad', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 0)
#------------------------------
#---------ADG------------------
n_qubits = 3
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_adg
S = Simulate('adg', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 21)
#---------BF------------------
n_qubits = 2
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_bf
S = Simulate('bf', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True,pi/2,pi/2)
#---------BPF------------------
n_qubits = 2
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_bpf
S = Simulate('bpf', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 0)
#---------d------------------
n_qubits = 3
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_d
S = Simulate('d', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 0)
#---------hw------------------
# n_qubits = 2
# d_rho_A = 2
# list_p = np.linspace(0.001,60,21)
# epochs = 120
# step_to_start = 80
# rho_AB = QCH.rho_AB_ad
# S = Simulate('ad', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 0)
#---------l------------------
n_qubits = 2
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_l
S = Simulate('l', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 0)
#---------PF------------------
n_qubits = 2
d_rho_A = 2
list_p = np.linspace(0.001,60,21)
epochs = 120
step_to_start = 80
rho_AB = QCH.rho_AB_pf
S = Simulate('pf', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
S.run_calcs_noMarkov(True, pi/2, 0)
#------------------------------