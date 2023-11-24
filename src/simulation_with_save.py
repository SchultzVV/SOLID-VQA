from torch.autograd import Variable
import pennylane as qml
from qiskit import *
from qiskit import Aer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import tensor
from numpy import pi

import os
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
import sys
sys.path.append('runtime-qiskit')
sys.path.append('src')
#sys.path.append('src')
import pickle
import ipywidgets as widgets
from IPython.display import display
#from src.pTrace import pTraceR_num, pTraceL_num
#from src.coherence import coh_l1
#from src.kraus_maps import QuantumChannels as QCH
#from src.theoric_channels import TheoricMaps as tm

from pTrace import pTraceR_num, pTraceL_num
from coherence import coh_l1
from kraus_maps import QuantumChannels as QCH
from kraus_maps import get_list_p_noMarkov
from theoric_channels import TheoricMaps as tm
from numpy import cos, sin, sqrt, pi, exp

class Simulate(object):

    def __init__(self, map_name, n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB):
        self.list_p = list_p
        self.epochs = epochs
        self.step_to_start = step_to_start
        self.rho_AB = rho_AB
        self.coerencias_R = [] # Essa é a certa
        self.map_name = map_name
        self.coerencias_L = []
        self.n_qubits = n_qubits
        self.d_rho_A = d_rho_A
        self.depht = n_qubits +1
   
    def get_device(self):
        device = qml.device('qiskit.aer', wires=self.n_qubits, backend='qasm_simulator')
        return device


    def prepare_rho(self, theta, phi, p, gamma=None):
        if gamma == None:
            rho = self.rho_AB(theta, phi, p)
            return rho
        else:
            rho = self.rho_AB(theta, phi, p, gamma)
            return rho

    def prepare_target_op(self, theta, phi, p, gamma):
        QCH.get_target_op(self.prepare_rho(theta, phi, p))

    def plot_theoric_map(self, theta, phi):
        a = tm()
        descript = 'isometria'
        # x = np.linspace(0,1,300)
        a.plot_theoric(self.list_p, self.map_name, theta, phi, descript)

    def read_data(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def general_vqacircuit_penny(self, params, n_qubits, depht=None):
        if depht == None:
            depht = self.n_qubits+1
        n = 3*self.n_qubits*(1+depht)
        #params = random_params(n)
        #params = [i for i in range(0,n)]
        #print(len(params))
        device = self.get_device()
        @qml.qnode(device, interface="torch")
        def circuit(params, M=None):
            w = [i for i in range(self.n_qubits)]
            aux = 0
            if self.n_qubits == 1:
                for j in range(depht+1):
                    qml.RX(params[aux], wires=0)
                    aux += 1
                    qml.RY(params[aux], wires=0)
                    aux += 1
                    qml.RZ(params[aux], wires=0)
                    aux += 1
                return qml.expval(qml.Hermitian(M, wires=w))
            for j in range(depht+1):
                for i in range(self.n_qubits):
                    qml.RX(params[aux], wires=i)
                    aux += 1
                    qml.RY(params[aux], wires=i)
                    aux += 1
                    qml.RZ(params[aux], wires=i)
                    aux += 1
                if j < depht:
                    for i in range(self.n_qubits-1):
                        qml.CNOT(wires=[i,i+1])
            return qml.expval(qml.Hermitian(M, wires=w))
        return circuit, params
    
    def start_things(self, depht):
        n = 3*self.n_qubits*(1+depht)
        params = np.random.normal(0,np.pi/2, n)
        params = Variable(tensor(params), requires_grad=True)
        return self.n_qubits, params, depht, n

    def cost(self, circuit, params, target_op):
        L = (1-(circuit(params, M=target_op)))**2
        return L

    def fidelidade(self, circuit, params, target_op):
        return circuit(params, M=target_op).item()

    def train(self, epocas, circuit, params, target_op, pretrain, pretrain_steps):
        opt = torch.optim.Adam([params], lr=0.1)
        best_loss = 1*self.cost(circuit, params, target_op)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_params = 1*params
        f=[]
        if pretrain:
            for start in range(pretrain_steps):
                opt.zero_grad()
                loss = self.cost(circuit, params, target_op)
                #print(epoch, loss.item())
                loss.backward()
                opt.step()
                if loss < best_loss:
                    best_loss = 1*loss
                    best_params = 1*params

        for epoch in range(epocas):
            opt.zero_grad()
            loss = self.cost(circuit, params, target_op)
            #print(epoch, loss.item())
            loss.backward()
            opt.step()
            if loss < best_loss:
                best_loss = 1*loss
                best_params = 1*params
            z = self.fidelidade(circuit, best_params, target_op)
            f.append(z)
        return best_params, f


    def general_vqacircuit_qiskit(self, n_qubits, params):
        #n = 3*self.n_qubits*(1+depht) # n=len(params)
        depht = int(len(params)/(3*self.n_qubits)-1)
        qr = QuantumRegister(self.n_qubits); qc = QuantumCircuit(qr)
        aux = 0
        for j in range(depht+1):
            for i in range(self.n_qubits):
                qc.rx(params[aux],i)
                aux += 1
                qc.ry(params[aux],i)
                aux += 1
                qc.rz(params[aux],i)
                aux += 1
            if j < depht:
                for i in range(self.n_qubits-1):
                    qc.cnot(i,i+1)
        return qc, qr

    def prepare_optmize(self, epochs, n_qubits, circuit, params, target_op, pretrain, step_to_start,theta, phi):
        for i in range(epochs):

            for p in self.list_p:
                # _, params, _, _ = self.start_things(self.depht)
                print(f'Canal de {self.map_name}, p = {count}, de {len(self.list_p)}')
                count += 1
                circuit, _ = self.general_vqacircuit_penny(params, self.n_qubits, self.depht)

                # defina o estado a ser preparado abaixo
                #------------------------------------------------------------
                target_op = QCH.get_target_op(self.prepare_rho(theta, phi, p))
                #------------------------------------------------------------

                # self.qc, self.qr, params, f = self.prepare_optmize(self.epochs, self.n_qubits, circuit, params, target_op, pretrain, self.step_to_start)

                self.qc, self.qr, params, f = self.optmize(1, n_qubits, circuit, params, target_op, pretrain, step_to_start)

    def optmize(self, epochs, n_qubits, circuit, params, target_op, pretrain, pretrain_steps):
        best_params, f = self.train(epochs, circuit, params, target_op, pretrain, pretrain_steps)
        parametros = best_params.clone().detach().numpy()
        qc, qr = self.general_vqacircuit_qiskit(self.n_qubits, parametros)
        best_params = Variable(tensor(parametros), requires_grad=True)
        return qc, qr, best_params, f

    def tomograph(self):
        if self.d_rho_A == 4:
            qstc = state_tomography_circuits(self.qc, [self.qr[0],self.qr[1],self.qr[2],self.qr[3]])
        elif self.d_rho_A == 2:
            qstc = state_tomography_circuits(self.qc, [self.qr[0],self.qr[1]])
        nshots = 8192
        job = execute(qstc, Aer.get_backend('qasm_simulator'), shots=nshots)
        qstf = StateTomographyFitter(job.result(), qstc)
        rho = qstf.fit(method='lstsq')
        return rho

    def results(self, rho, coerencias_R, coerencias_L):
        rho_R = pTraceR_num(self.d_rho_A, self.d_rho_A, rho)
        rho_L = pTraceL_num(self.d_rho_A, self.d_rho_A, rho)
        coh_R = coh_l1(rho_R)
        coh_L = coh_l1(rho_L)
        coerencias_R.append(coh_R)
        coerencias_L.append(coh_L)

        return coerencias_L, coerencias_R
    
    def results_hw(self, rho, coerencias_R, coerencias_L):
        rho_R = pTraceR_num(4,4,rho)
        rho_L = pTraceL_num(4,4,rho)
        coh_R = coh_l1(rho_R)
        coh_L = coh_l1(rho_L)
        coerencias_R.append(coh_R)
        coerencias_L.append(coh_L)

        return coerencias_L, coerencias_R
    
    def name_changer(self, map_name: any):
        
        if map_name == 'bpf':
            return 'bit-phase-flip'
        if map_name == 'ad':
            return 'amplitude-damping'
        if map_name == 'bf':
            return 'bit-flip'
        if map_name == 'pf':
            return 'phase-flip'
        if map_name == 'pd':
            return 'phase-damping'
        if map_name == 'd':
            return 'depolarizing'
        if map_name == 'adg':
            return 'generalized-amplitude-damping'
        if map_name == 'l':
            return 'Lorentz'
        if map_name == 'hw':
            return 'Heisenberg Weyl-dephasing'

    def plots(self, list_p, coerencias_L):
        print(list_p)
        print(len(coerencias_L))
        plt.scatter(list_p,coerencias_L,label='simulação')
        plt.xlabel(' p ')
        plt.ylabel(' Coerência ')
        plt.legend(loc=0)
        plt.savefig(f'figures/automatic/{self.map_name}.png')
        plt.show()

    def plots_markov(self, list_p, coerencias_L, theta, phi):
        print(list_p)
        print(len(coerencias_L))
        x = [i for i in range(len(coerencias_L))]
        plt.scatter(list_p, coerencias_L, label='simulação')
        plt.xlabel(' t ')
        plt.ylabel(' Coerência ')
        plt.legend(loc=0)
        plt.savefig(f'noMarkov/figures/automatic/{self.map_name}.png')
        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        fancy_name = self.name_changer(self.map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle =$ " + psi
        if self.map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.title(m,usetex=True)
        plt.suptitle(fancy_name)
        if self.map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('t')
        plt.show()


    #def run_calcs(self, save, theta, phi):#, gamma=None):
    def run_calcs(self, save, theta, phi):#, gamma=None):
        #coerencias_R = []
        coerencias_L = []
        pretrain = True
        count = 0
        _, params, _, _ = self.start_things(self.depht)
        for p in self.list_p:
            print(f'{count} de {len(self.list_p)}')
            count += 1
            circuit, _ = self.general_vqacircuit_penny(params, self.n_qubits, self.depht)

            # defina o estado a ser preparado abaixo
            #------------------------------------------------------------
            target_op = QCH.get_target_op(self.prepare_rho(theta, phi, p))
            #------------------------------------------------------------

            self.qc, self.qr, params, f = self.optmize(self.epochs, self.n_qubits, circuit, params, target_op, pretrain, self.step_to_start)
            pretrain = False
            data = {'map_name': self.map_name,
                    'params': params,
                    'epochs': self.epochs,
                    'theta': theta,
                    'phi': phi,
                    'p': p}
            print(data)
            if save:
                filename = f'data/{self.map_name}/paramsP_{p:.2f}theta_{theta:.2f}_phi{phi:.2f}.pkl'
                if os.path.isfile(filename):
                    print(f'O arquivo {filename} já existe. Não salve novamente.')
                    with open(filename, 'wb') as f:
                        pickle.dump(data, f)
                    pass

                else:
                    with open(filename, 'wb') as f:
                        pickle.dump(data, f)
            rho = self.tomograph()
            #print(rho)
            self.coerencias_L, self.coerencias_R = self.results(rho, self.coerencias_R, coerencias_L)
        mylist = [self.coerencias_L, self.coerencias_R]
        if save:
            with open(f'data/{self.map_name}/coerencia_L_e_R.pkl', 'wb') as f:
                pickle.dump(mylist, f)
        # if self.map_name == 'hw':
        #    pass
        # else:
        #    self.plot_theoric_map(theta, phi)
        self.plot_theoric_map(theta, phi)
        self.plots(self.list_p, self.coerencias_L)
    
    def run_calcs_noMarkov(self, save, theta, phi, continuous_coh):#, gamma=None):
        #coerencias_R = []
        print(self.map_name)
        caminho = f'data/noMarkov/{self.map_name}/coerencia_L_e_R__list_t.pkl'
        if continuous_coh:
            try:
                coerencias_L = self.read_data(caminho)[0]
                if len(coerencias_L) < len(self.list_p):
                    print(len(self.list_p))
                    print(len(coerencias_L))
                    faltam = len(self.list_p)-len(coerencias_L)
                    print('ainda faltam', faltam)
                    self.list_p = self.list_p[len(coerencias_L):]
                    print(len(self.list_p))
            except:
                coerencias_L =[]    
        else:
            coerencias_L =[]
        # sys.exit()
        print('list_t = ', self.list_p)
        x = self.list_p
        #self.list_p = get_list_p_noMarkov(self.list_p,'Bellomo')
        self.list_p = get_list_p_noMarkov(self.list_p,'Ana')
        self.list_p = [i/max(self.list_p) for i in self.list_p]
        print('list_t = ', self.list_p)
        pretrain = True
        count = 0
        _, params, _, _ = self.start_things(self.depht)
        # self.qc, self.qr, params, f = self.prepare_optmize(80, self.n_qubits, circuit, params, target_op, pretrain, self.step_to_start,theta, phi)

        for p in self.list_p:
            # _, params, _, _ = self.start_things(self.depht)
            print(f'Canal de {self.map_name}, p = {count}, de {len(self.list_p)}')
            count += 1
            circuit, _ = self.general_vqacircuit_penny(params, self.n_qubits, self.depht)

            # defina o estado a ser preparado abaixo
            #------------------------------------------------------------
            target_op = QCH.get_target_op(self.prepare_rho(theta, phi, p))
            #------------------------------------------------------------
            

            self.qc, self.qr, params, f = self.optmize(self.epochs, self.n_qubits, circuit, params, target_op, pretrain, self.step_to_start)
            pretrain = False
            data = {'map_name': self.map_name,
                    'params': params,
                    'epochs': self.epochs,
                    'theta': theta,
                    'phi': phi,
                    'p': p}
            print(data)
            if save:
                #filename = f'noMarkov/data/{self.map_name}/state/paramsP_{p:.2f}theta_{theta:.2f}_phi{phi:.2f}_{self.epochs}.pkl'
                filename = f'data/noMarkov/{self.map_name}/state/paramsP_{p:.2f}theta_{theta:.2f}_phi{phi:.2f}_{self.epochs}.pkl'
                if os.path.isfile(filename):
                    print(f'O arquivo {filename} já existe. Não salve novamente.')
                    pass
                else:
                    with open(filename, 'wb') as f:
                        pickle.dump(data, f)
            rho = self.tomograph()
            #print(rho)
            self.coerencias_L, self.coerencias_R = self.results(rho, self.coerencias_R, coerencias_L)
            mylist = [self.coerencias_L, self.coerencias_R, self.list_p]
            if save:
                with open(caminho, 'wb') as f:
                    pickle.dump(mylist, f)
        # self.plot_theoric_map(theta, phi)
        self.plots_markov(self.list_p, self.coerencias_L, theta, phi)
        print('Deu, acabou.')

    def rho_from_qc(self, best_params):
        params = best_params.clone().detach().numpy()
        self.qc, self.qr = self.general_vqacircuit_qiskit(self.n_qubits, params)
        #self.qc, self.qr = self.general_vqacircuit_penny(best_params, self.n_qubits, self.depht)
        rho = self.tomograph()
        print(rho)
        self.coerencias_L, self.coerencias_R = self.results(rho, self.coerencias_R, self.coerencias_L)
        print(self.coerencias_R)
        return rho

    def reload_rho(self, map_name, markovianity):
        if markovianity:
            pasta = f'data/{map_name}/state/'  # Substitua pelo caminho da sua pasta
            arquivos_pkl = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('.pkl')]
            for arquivo in arquivos_pkl:
                path = pasta+arquivo
                data = self.read_data(path)
                best_params = data['params']
                print(best_params)
                rho = self.rho_from_qc(best_params)
                self.coerencias_L, self.coerencias_R = self.results(rho, self.coerencias_R, coerencias_L) #7 set
                print(self.coerencias_L)
                print(self.coerencias_R)
                
                
        else:
            pasta = f'noMarkov/data/{map_name}/state'
            arquivos_pkl = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('.pkl')]
            for arquivo in arquivos_pkl:
                path = pasta+arquivo
                with open(path, 'rb') as file:
                    # Use pickle.load() para desserializar o objeto
                    best_params = pickle.load(file)
                #params = ler com pickle
                coerencias_L = []
                count = 0
                print(best_params['params'])
                params = best_params['params'].clone().detach().numpy()
                self.qc, self.qr = self.general_vqacircuit_qiskit(self.n_qubits, params)
                #self.qc, self.qr = self.general_vqacircuit_penny(best_params, self.n_qubits, self.depht)
                rho = self.tomograph()
                print(rho)
                return rho
            # print(arquivo)
            # print(type(arquivo))
    
    def plot_bloch(self , density_matrix):
        from qutip import Bloch, Qobj, tensor, sigmax, sigmay, sigmaz
        density_matrix = np.array([[0.69356156+0.j, 0.26884403-0.16545896j, 0.23831009-0.18374915j, 0.14206474-0.01111998j],
                           [0.26884403+0.16545896j, 0.14528759+0.j, 0.13498041-0.01511379j, 0.05790133+0.0295653j],
                           [0.23831009+0.18374915j, 0.13498041+0.01511379j, 0.13185252+0.j, 0.05162894+0.03391243j],
                           [0.14206474+0.01111998j, 0.05790133-0.0295653j, 0.05162894-0.03391243j, 0.02929833+0.j]])
        density_matrix_qobj = Qobj(density_matrix)
        # Define Pauli operators for the 2-qubit system
        sigma_x = tensor(sigmax(), sigmax())
        sigma_y = tensor(sigmay(), sigmay())
        sigma_z = tensor(sigmaz(), sigmaz())

        # Create a Bloch sphere object and add the density matrix with the corresponding Pauli operators
        bloch_sphere = Bloch()
        bloch_sphere.add_states(density_matrix_qobj, 'dm', [sigma_x, sigma_y, sigma_z])

        # Plot the Bloch sphere
        bloch_sphere.show()

        # Save the plot to a file (optional)
        # bloch_sphere.save('bloch_sphere.png')

        # Show the plot
        plt.show()

    def worthed_plot_bloch():
        import numpy as np
        import matplotlib.pyplot as plt
        from qutip import Bloch, basis

        # Define your state vector
        state_vector = np.array([1.8415, -0.8333, -1.1572, 0.8130, -0.0611, 0.4253, -0.2375, -0.6908,
                                 -0.0169, 2.5574, 0.9711, -1.3900, 0.0281, 0.5020, -1.1698, 0.4683,
                                 -0.6323, -1.5013, 1.2044, 2.9128, -0.5611, -0.7137, -2.4128, -3.0238],
                                dtype=np.complex128)

        # Determine the number of qubits
        num_qubits = int(np.log2(len(state_vector)))

        # Create a Bloch sphere object
        bloch_sphere = Bloch()

        # Add the state vector amplitudes to the Bloch sphere
        for i in range(2 ** num_qubits):
            amplitude = state_vector[i]
            bloch_sphere.add_vectors([np.real(amplitude), np.imag(amplitude), 0])

        # Plot the Bloch sphere
        bloch_sphere.show()

        # Save the plot to a file (optional)
        # bloch_sphere.save('bloch_sphere.png')

        # Show the plot
        plt.show()


    def run_sequential_bf(self, phis):
        for i in phis:
            self.run_calcs(True, pi/2, i)


def main():
  
    n_qubits = 2
    d_rho_A = 2
    theta = pi/2
    phi = 0
    list_p = np.linspace(0.01,1000,21)
    markovianity = False
    saving = True
    epochs = 60
    step_to_start = 80
    append_data = False
    rho_AB = QCH.rho_AB_ad
    S = Simulate('ad', n_qubits, d_rho_A, list_p, epochs, step_to_start, rho_AB)
    #rho = np.array(S.reload_rho('pd', markovianity))
    #S.plot_bloch(rho)
    #print(rho)
    #sys.exit()
    if not markovianity:
        S.run_calcs_noMarkov(saving, theta, phi, append_data)
    if markovianity:
        S.run_calcs(saving, theta, phi)
    
    #phis = [0,pi,pi/1.5,pi/2,pi/3,pi/4,pi/5]
    #S.run_sequential_bf(phis)
    #plt.legend(loc=1)
    #plt.show()

if __name__ == "__main__":
    main()

#from src.theoric_channels import TheoricMaps as TM
#plot_theoric = TM.theoric_rho_A_bpf
#rho_AB = QCH.rho_AB_bpf
#n_qubits = 2
#list_p = np.linspace(0,1,5)
#epochs = 1
#step_to_start = 1
#
#S = Simulate('bpf/ClassTest', n_qubits, list_p, epochs, step_to_start, rho_AB, plot_theoric)
#S.run_calcs()
#print(S)