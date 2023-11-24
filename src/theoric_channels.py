from sympy import cos, sin, pi, Matrix, Symbol, exp, print_latex, simplify
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from math import sqrt
from coherence import coh_l1
import pickle
import sys, os
from kraus_maps import get_list_p_noMarkov
#from simulation_with_save import Simulate as S
from qutip import Bloch, basis

#sys.path.append('runtime-qiskit')
sys.path.append('src')

class TheoricMaps():
    def __init__(self):
        self.theta = Symbol('theta',real=True)
        self.phi = Symbol('phi',real=True)
        self.gamma = Symbol('gamma',real=True, positive=True)
        self.p = Symbol('p',real=True, positive=True)
        #self.theta = theta
        #self.phi = phi
        #self.gamma = gamma
        #self.p = p
        #self.path_save = f"result_{camera.split('/')[-1]}"\
        #    .replace(".mp4", ".csv")
    
    def read_data(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def name_changer(map_name):
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

    def map_choser(self,map_name):
        list_of_maps = ['bpf','ad','bf','pf','pd','adg','d','l','hw']#,'new']
        list_of_functions = [self.theoric_rho_A_bpf, self.theoric_rho_A_ad,
                             self.theoric_rho_A_bf, self.theoric_rho_A_pf,
                             self.theoric_rho_A_pd, self.theoric_rho_A_adg,
                             self.theoric_rho_A_d, self.theoric_rho_A_l,
                             self.theoric_rho_A_hw
                            ]
            #self.theoric_rho_A_H,
            #self.theoric_rho_A_ad3   ]
        if map_name in list_of_maps:
            #print(list_of_maps.index(map_name))
            return list_of_functions[list_of_maps.index(map_name)]
        
    def map_choser_coh(self,map_name):
        list_of_maps = ['bpf','ad','bf','pf','pd','adg','d','l','hw']#,'new']
        list_of_functions = [self.bpf, self.ad,
                             self.bf, self.pf,
                             self.pd, self.adg,
                             self.d, self.l,
                             self.hw
                            ]
            #self.theoric_rho_A_H,
            #self.theoric_rho_A_ad3   ]
        if map_name in list_of_maps:
            #print(list_of_maps.index(map_name))
            return list_of_functions[list_of_maps.index(map_name)]



    # def coh_l1(self,rho):  # normalized to [0,1]
        # d = rho.shape[0]
        # coh = 0.0
        # for j in range(0, d-1):
            # for k in range(j+1, d):
                # coh += math.sqrt((rho[j][k].real)**2.0 + (rho[j][k].imag)**2.0)
        # return 2.0*coh/(d-1)
    
    def coh_l1(self, rho):
        d = rho.shape[0]; C = 0
        for j in range(0,d-1):
            for k in range(j+1,d):
                C += abs(rho[j,k])
        return 2*C
    
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
    
    def print_state(self):
        
        return print_latex(self.coherence(self.theoric_rho_A_adg(self.theta,self.phi, self.p)))

    def theoric_rho_A_ad(self,theta, phi, p):
        state = Matrix([[p*(sin(theta/2)**2)+(cos(theta/2)**2),
                        (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))],[
                        (sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2)),
                        ((1-p)*sin(theta/2)**2)]])
        return state

    def theoric_rho_A_bf(self, theta, phi, p):
        state = Matrix([[(1-p)*((cos(theta/2))**2) + p*((sin(theta/2))**2),
                        (((exp(-1j*phi))+(2j*p*sin(phi)))*sin(theta/2)*cos(theta/2))],[
                        (((exp(1j*phi))-(2j*p*sin(phi)))*sin(theta/2)*cos(theta/2)),
                        (1-p)*(sin(theta/2)**2)+p*(cos(theta/2)**2)]])
        return state
    
    def theoric_rho_A_bpf(self, theta, phi, p):
        state = Matrix([[((1-p)*(cos(theta/2))**2)+p*(sin(theta/2)**2),
                        ((1-p)*exp(-1j*phi)*cos(theta/2)*sin(theta/2))-(p*exp(1j*phi)*cos(theta/2)*sin(theta/2))],[
                        ((1-p)*exp(1j*phi)*cos(theta/2)*sin(theta/2))-(p*exp(-1j*phi)*cos(theta/2)*sin(theta/2)),
                        ((1-p)*sin(theta/2)**2)+p*cos(theta/2)**2]])
        return state

    def theoric_rho_A_pd(self, theta, phi, p):
        state = Matrix([[(cos(theta/2)**2),
                        (sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))],[
                        (sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2)),
                        (sin(theta/2)**2)]])
        return state

    def theoric_rho_A_pf(self, theta, phi, p):
        state = Matrix([[(cos(theta/2))**2,
                        ((1-2*p)*exp(-1j*phi)*sin(theta/2)*cos(theta/2))],[
                        ((1-2*p)*exp(1j*phi)*sin(theta/2)*cos(theta/2)),
                        sin(theta/2)**2]])
        return state

    def theoric_rho_A_d(self, theta, phi, p):
        state = Matrix([[(p/2)*(sin(theta/2))**2+(1-p/2)*(cos(theta/2))**2,
                        ((1-p)*exp(-1j*phi)*sin(theta/2)*cos(theta/2))],[
                        ((1-p)*exp(1j*phi)*sin(theta/2)*cos(theta/2)),
                        ((1-p/2)*(sin(theta/2)**2))+(p/2)*(cos(theta/2))**2
                        ]])
        return state

    def theoric_rho_A_l(self, theta, phi, p):
        state = Matrix([[(cos(p/2)**2)*(cos(theta/2)**2)+(sin(p/2)**2)*(sin(theta/2)**2),
                        (cos(p/2)**2)*exp(-1j*phi)*cos(theta/2)*sin(theta/2)-(sin(p/2)**2)*exp(1j*phi)*cos(theta/2)*sin(theta/2)],[
                        (sin(p/2)**2)*exp(-1j*phi)*cos(theta/2)*sin(theta/2)+(cos(p/2)**2)*exp(1j*phi)*cos(theta/2)*sin(theta/2),
                        (sin(p/2)**2)*(cos(theta/2)**2)+(cos(p/2)**2)*(sin(theta/2)**2)]])
        return state
    @staticmethod
    def theoric_rho_A_adg(theta, phi, p):
        N = 0.5

        state = Matrix([[((1-N)*cos(theta/2)+p*(1-N)*(sin(theta/2))**2+N*(1-p)*cos(theta/2)),
                        sqrt(1-p)*exp(-1j*phi)*sin(theta/2)*cos(theta/2)],[
                        sqrt(1-p)*exp(1j*phi)*sin(theta/2)*cos(theta/2), #|010\rangle
                        ((1-p)+N)*sin(theta/2)**2+p*N*cos(theta/2) #|111\rangle)
                       ]])

        #state = Matrix([[(p*(1-N)*((sin(theta/2))**2)+N*(1-p)*cos(theta/2)),
        #                 sqrt(1-p)*exp(-1j*phi)*sin(theta/2)*cos(theta/2)],[
        #                 2*sqrt(1-p)*exp(1j*phi)*sin(theta/2)*cos(theta/2), #|010\rangle
        #                 ((1-p)+N)*sin(theta/2)**2+p*N*cos(theta/2) #|111\rangle)
        #                ]])
        return state
    
    @staticmethod
    def theoric_rho_A_hw2(theta, phi, p):
        N = 0.5
        z1 = (1j*sqrt(3)-1)/2
        z2 = (-1j*sqrt(3)-1)/2
        state = Matrix([[1/3,(3*p-1)/6,(3*p-1)/6,0
                        ],[(3*p-1)/6, 1/3, (3*p-1)/6,0],[(3*p-1)/6,(3*p-1)/6,1/3,0],
                        [0,0,0,0]])
        print(state)
        return state

    @staticmethod
    def theoric_rho_A_hw2(theta, phi, p):
        A = (2*p-1)/(3)
        B = (3*p-1)/(6)
        state = Matrix([[1/3, (p/3 + (1 - p*(-1j*sqrt(3) - 1))/12 + (1 - p*(1j*sqrt(3) + 1))/12), (p/3 - 1 - p*(-1j*sqrt(3) + 1)/12 + (1 - p*(-1j*sqrt(3) - 1))/6)],
                    [(p/3 - 1 - p*(1j*sqrt(3) - 1)/12 - (1 - p*(1j*sqrt(3) + 1))/12), (p/3 + (1 - p)/6 - (1 - p*(2*1j*sqrt(3) - 1))/12), (p/3 + (1 - p*(-1j*sqrt(3) - 1))/12 - (1 - p*(1 - 1j*sqrt(3)))/12)],
                    [(p/3 - (1 - p*(1j*sqrt(3) + 1))/12 + (1 - p*(1j*sqrt(3) - 1))/12), (p/3 - (1 - p*(1 - 1j*sqrt(3)))/12 - (1 - p)/6), (p/3 - (1 - p)/6 + (1 - p)/6)]])
        # print(state)
        return state
    
    def theoric_rho_A_hw(self, theta, phi, p):
        A = (2*p-1)/(3)
        B = (3*p-1)/(6)
        state = Matrix([[1/3, B, B, 0],
                    [B, A, B, 0],
                    [B, B, A, 0],
                    [0, 0, 0, 0]])
        # print(state)
        return state
    
    def theoric_rho_A_hw2(self, theta, phi, p):
        p0 = p
        p1 = (1-p)/2
        p2 = (1-p)/2
        A = (p0+p1+p2)/3
        
        state = Matrix([[A,(1/3)*(p0+p1*exp(-2*1j*pi/3)+p2*exp(-4*1j*pi/3)), (1/3)*(p0+p1*exp(-4*1j*pi/3)+p2*exp(8*1j*pi/3)), 0],
                    [(1/3)*(p0+p1*exp(2*1j*pi/3)+p2*exp(4*1j*pi/3)), A, (1/3)*(p0+p1*exp(-2*1j*pi/3)+p2*exp(-4*1j*pi/3)), 0],
                    [(1/3)*(p0+p1*exp(4*1j*pi/3)+p2*exp(8*1j*pi/3)), (1/3)*(p0+p1*exp(2*1j*pi/3)+p2*exp(4*1j*pi/3)), A, 0],
                    [0, 0, 0, 0]])
        # print(state)
        return state
    

    def plot_storaged(self, list_p, map_name, lambd, markovianity):
        #path = f'../data/{map}/{map}-coherences.pkl'
        if markovianity:
            x2 = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]))
            print(markovianity)
            path = f'data/{map_name}/coerencia_L_e_R.pkl'
            coh_l = self.read_data(path)[0]#.detach().numpy()
            if map_name == 'l':
            #    plt.plot(np.linspace(0,1.6,len(coh_l)),coh_l,label='Simulação '+map_name+' Makorv')
               plt.scatter(np.linspace(0, 1.6, len(coh_l)), coh_l, label='simulação')
            else:
                plt.scatter(np.linspace(0, 1, len(coh_l)), coh_l, label='Simulação Markoviana')
                # plt.plot(np.linspace(0,1,len(coh_l)),coh_l,label=map_name)
        else:
            path = f'data/noMarkov/{map_name}/coerencia_L_e_R.pkl'
            # path = f'data/noMarkov/{map_name}/coerencia_L_e_R__list_t.pkl'
            x2 = np.array([self.non_markov_t_Ana(lambd, i) for i in list_p])
            coh_l = self.read_data(path)[0]#.detach().numpy()
            #print(self.read_data(path))
            print(len(coh_l))
            #x2 = np.linspace(0.01, 1000, len(coh_l))
            if map_name == 'l':
                # plt.scatter(np.linspace(0,1.6,len(coh_l)),coh_l,label='Simulação N - Makorv')
                plt.scatter(np.linspace(0.01, 1000, len(coh_l)), coh_l, label='simulação')
            else:
                # plt.scatter(np.linspace(0,1,len(coh_l)),coh_l,label=map_name)
                # x2 = np.linspace(0, 1000, len(coh_l))
                # xa = np.array([self.non_markov_t_Ana(0.001, i) for i in x2])
                plt.plot(list_p,coh_l)#, label=map_name)#+' N - Makorv')
                plt.scatter(list_p, coh_l, label=map_name,color='red')
                # plt.scatter(xa,coh_l,label='Simulação não Markoviana')
                # plt.xscale('log')
                # plt.scatter(xa,coh_l,label=map_name+' N - Makorv')
                # plt.plot(xa,coh_l)#,label='x2')
                # plt.scatter(xa,coh_l,label='xa')
        print(path)
        

    
    def plot_storaged2(self, map_name, markovianity):
        #path = f'../data/{map}/{map}-coherences.pkl'
        if markovianity:
            try:
                path = f'data/{map_name}/coerencia_L_e_R.pkl'
                coh_l = self.read_data(path)[0]#.detach().numpy()
            except:

                path = f'data/{map_name}/ClassTestcasa.pkl'
                coh_l = self.read_data(path)[0]#.detach().numpy()
        else:
            path = f'noMarkov/data/{map_name}/coerencia_L_e_R.pkl'
            coh_l = self.read_data(path)[0]#.detach().numpy()
        plt.plot(np.linspace(0,1,len(coh_l)),coh_l,label=f'{map_name}')

    def reload_rho(self, map_name, markovianity):
        if markovianity:
            pasta = f'data/{map_name}/state'  # Substitua pelo caminho da sua pasta
        else:
            pasta = f'noMarkov/data/{map_name}/state'
        arquivos_pkl = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith('.pkl')]

        for arquivo in arquivos_pkl:
            print(arquivo)
            print(type(arquivo))

    def plot_theoric(self, list_p, map_name, theta, phi, descript):
        cohs = []
        if map_name == 'l':
            list_p = np.linspace(0,pi/2,len(list_p))
        for pp in list_p:
            rho = self.map_choser(map_name)(theta,phi,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            cohs.append(coh)
        #m = f'Estado inicial, theta =  {str(theta)[0:4]}, phi = {str(phi)[0:4]}'
        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        fancy_name = self.name_changer(map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle =$ " + psi
        if map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            # m0 = r"Estado inicial (bpf) $|\psi(\theta,\phi)\rangle = |\psi(\pi/2,\pi/2)\rangle, $"
            # m1 = r"Estado inicial (bf - pf) $|\psi(\theta,\phi)\rangle =$ " + psi
            # m2 = r"estado inicial (hw) $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$"
            # m = m0+m2
            # plt.suptitle(m1)
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.suptitle(fancy_name)
        plt.title(m,usetex=True)
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')
        plt.ylabel('coerência')
        # plt.scatter(list_p,cohs,label=descript)
        plt.plot(list_p,cohs,label=descript)
        plt.title(m)

    def plot_theoric_n_Markov(self, list_p, map_name, theta, phi, lambd, descript):
        cohs = []
        # x2 = np.linspace(0.01,1000,len(list_p))
        # xa = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]))
        xa = np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]) # mode sem ordenar
        # xa = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p])) # mode ordenado
        # xa = np.array([self.non_markov_t_Ana(lambd, i) for i in x2])
        if map_name != 'l':
            xa = [i/max(xa) for i in xa]
        for pp in xa:
            rho = self.map_choser(map_name)(theta,phi,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            # print(coh)
            cohs.append(coh)
        #m = f'Estado inicial, theta =  {str(theta)[0:4]}, phi = {str(phi)[0:4]}'
        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        lamb_info = r"$\lambda = $"+str(lambd)
        fancy_name = self.name_changer(map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle = $ " + psi
        if map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            # m0 = r"Estado inicial (bpf) $|\psi(\theta,\phi)\rangle = |\psi(\pi/2,\pi/2)\rangle, $"
            # m1 = r"Estado inicial (bf - pf) $|\psi(\theta,\phi)\rangle =$ " + psi
            # m2 = r"estado inicial (hw) $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$"
            # m = m0+m2
            # plt.suptitle(m1)
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.suptitle(fancy_name)
        plt.title(m,usetex=True)
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')
        plt.ylabel('coerência')
        plt.scatter(list_p,cohs,label=descript)
        # plt.plot(list_p, cohs, label=descript)
        plt.title(m)

    def theoric_plot(self, list_p, map_name, theta, phi, lambd, descript, Markovianity):
        cohs = []
        # x2 = np.linspace(0.01,1000,len(list_p))
        if Markovianity:
            xa = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]))
        else:
            xa = np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]) # mode sem ordenar
        # xa = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p])) # mode ordenado
        # xa = np.array([self.non_markov_t_Ana(lambd, i) for i in x2])
        if map_name != 'l':
            xa = [i/max(xa) for i in xa]
        for pp in xa:
            rho = self.map_choser(map_name)(theta,phi,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            # print(coh)
            cohs.append(coh)
        #m = f'Estado inicial, theta =  {str(theta)[0:4]}, phi = {str(phi)[0:4]}'
        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        lamb_info = r"$\lambda = $ "+str(lambd)
        fancy_name = self.name_changer(map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle = $ " + psi
        if map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            # m0 = r"Estado inicial (bpf) $|\psi(\theta,\phi)\rangle = |\psi(\pi/2,\pi/2)\rangle, $"
            # m1 = r"Estado inicial (bf - pf) $|\psi(\theta,\phi)\rangle =$ " + psi
            # m2 = r"estado inicial (hw) $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$"
            # m = m0+m2
            # plt.suptitle(m1)
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.suptitle(fancy_name)
        plt.title(m,usetex=True)
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')
        plt.ylabel('coerência')
        # plt.scatter(list_p,cohs,label=descript+lamb_info) # usa o list_p
        # plt.scatter(x2, cohs, label=descript)
        plt.plot(list_p, cohs, label=descript+lamb_info)
        plt.title(m)

    def theoric_plot2(self, list_p, map_name, theta, phi, lambd, descript, Markovianity):
        cohs = []
        # x2 = np.linspace(0.01,1000,len(list_p))
        if Markovianity:
            xa = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]))
        else:
            xa = np.array([self.non_markov_t_Ana(lambd, i) for i in list_p]) # mode sem ordenar
        # xa = np.sort(np.array([self.non_markov_t_Ana(lambd, i) for i in list_p])) # mode ordenado
        # xa = np.array([self.non_markov_t_Ana(lambd, i) for i in x2])
        if map_name != 'l':

            xa = [i/max(list_p) for i in list_p]
        for pp in xa:
            rho = self.map_choser(map_name)(theta,phi,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            # print(coh)
            cohs.append(coh)
        #m = f'Estado inicial, theta =  {str(theta)[0:4]}, phi = {str(phi)[0:4]}'
        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        lamb_info = r"$\lambda = $"+str(lambd)
        fancy_name = self.name_changer(map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        if map_name == 'l':
            lamb_info = 'teórico'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle = $ " + psi
        if map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            # m0 = r"Estado inicial (bpf) $|\psi(\theta,\phi)\rangle = |\psi(\pi/2,\pi/2)\rangle, $"
            # m1 = r"Estado inicial (bf - pf) $|\psi(\theta,\phi)\rangle =$ " + psi
            # m2 = r"estado inicial (hw) $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$"
            # m = m0+m2
            # plt.suptitle(m1)
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.suptitle(fancy_name)
        plt.title(m,usetex=True)
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')
        plt.ylabel('coerência')
        # plt.scatter(list_p,cohs,label=descript)
        # plt.scatter(x2, cohs, label=descript)
        plt.plot(list_p, cohs, label=descript+lamb_info)
        plt.title(m)


    def plot_theoric_n_Markov_B(self, list_p, map_name, theta, phi, descript):
        # if map_name == 'l':
            # list_p = np.linspace(0,pi/2,len(list_p))
        cohs = []
        x = np.linspace(0,pi/2,len(list_p))
        x2 = np.linspace(0.01,1000,len(list_p))
        # xa = np.array([self.non_markov_t_Ana(0.01,i) for i in x2])
        xa = np.array([self.non_markov_t_Bellomo(0.01,i) for i in x2])
        if map_name != 'l':
            xa = [i/max(xa) for i in xa]
        for pp in xa:
            rho = self.map_choser(map_name)(theta,phi,pp)
            rho_numpy = np.array(rho.tolist(), dtype=np.complex64)
            coh = self.coh_l1(rho_numpy)
            # print(coh)
            cohs.append(coh)
        #m = f'Estado inicial, theta =  {str(theta)[0:4]}, phi = {str(phi)[0:4]}'
        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        fancy_name = self.name_changer(map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle =$ " + psi
        if map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            # m0 = r"Estado inicial (bpf) $|\psi(\theta,\phi)\rangle = |\psi(\pi/2,\pi/2)\rangle, $"
            # m1 = r"Estado inicial (bf - pf) $|\psi(\theta,\phi)\rangle =$ " + psi
            # m2 = r"estado inicial (hw) $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$"
            # m = m0+m2
            # plt.suptitle(m1)
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.suptitle(fancy_name)
        plt.title(m,usetex=True)
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')
        plt.ylabel('coerência')
        # plt.scatter(list_p,cohs,label=descript)
        # plt.scatter(x2, cohs, label=descript)
        plt.plot(x2, cohs, label=descript)
        plt.title(m)
    
    def plot_all_theoric_space(self,map):
        li = np.linspace(0,2*np.pi, 5)
        x1 = np.linspace(0,1,21)
        if map == 'l':
            x = np.linspace(0,2,21)
        for i in li:
            for k in li:
                self.plot_theoric(x1,map,theta=k,phi=i)
        lines = plt.gca().get_lines()
        labels = [line.get_label() for line in lines[:8]]
        plt.legend(labels=labels,loc=1)
        plt.show()
    
    def plot_bloch(self,state):
        from qutip import Bloch, basis
        num_qubits = int(np.log2(len(state)))
        state_ket = basis(2 ** num_qubits, 0)
        for i, amp in enumerate(state):
            state_ket += amp * basis(2 ** num_qubits, i)

        # Create a Bloch sphere object and add the state vector to it
        bloch_sphere = Bloch()
        bloch_sphere.add_states(state_ket)

        # Plot the Bloch sphere
        bloch_sphere.show()

        # Save the plot to a file (optional)
        # bloch_sphere.save('bloch_sphere.png')

        # Show the plot
        plt.show()


    def coherence(self, state):
        # Extrai os elementos do vetor de estado
        a11, a12, a21, a22 = state.tolist()[0]
        # Calcula as normas L2 dos elementos
        norm_a1 = sqrt(abs(a11)**2 + abs(a21)**2)
        norm_a2 = sqrt(abs(a12)**2 + abs(a22)**2)
        # Calcula o produto interno dos elementos
        inner_product = a11.conjugate()*a22 + a12.conjugate()*a21
        # Calcula a coerência
        coherence = abs(inner_product)/(norm_a1*norm_a2)
        # Retorna a expressão LaTeX da coerência
        return coherence
#------------------------------------------------------------------------------------------------------    

    def ad(self, p, theta, phi):
        p1 = abs(sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))
        p2 = abs(sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2))
        return p1+p2
        #theta = np.pi/2
        #phi = 0
        #return 
        return sqrt(2)*abs(np.sin(theta/2)) * abs(np.sqrt(p) * np.exp(1j * phi) + np.sqrt(1 - p) * np.exp(1j * phi))
        #return 2*sin(theta/2)*cos(theta/2)*sqrt(4*(p**2)*(cos(phi)**2)-4*p*(cos(phi)**2)+1) # bf
    
    def pd_uat(self, p, theta, phi):
        #theta = np.pi/2
        #phi = 0        
        return 2*np.sqrt(1-p)*np.sin(theta/2)*np.cos(theta/2)

    def pd(self, p, theta, phi):
        p1 = abs(sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))
        p2 = abs(sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2))
        return p1+p2

    def adg(self, p, theta, phi):
        # theta = np.pi/2
        # phi = 0        
        return (1-p)*np.sin(theta/2)*np.cos(theta/2)*np.sqrt(np.cos(phi)-np.sin(phi))

    def bf(self, p, theta, phi):
        return 0

    def pf(self, p, theta, phi):
        p1 = abs(sqrt(1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2))
        p2 = abs(sqrt(1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2))
        return p1+p2

    def bpf(self, p, theta, phi):
        # theta = np.pi/2
        # phi = np.pi/2
        p1 = abs((1-p)*cos(theta/2)*exp(-1j*phi)*sin(theta/2)-p*exp(1j*phi)*sin(theta/2)*cos(theta/2))
        p2 = abs((1-p)*cos(theta/2)*exp(1j*phi)*sin(theta/2)-p*exp(-1j*phi)*sin(theta/2)*cos(theta/2))
        return p1+p2
    
    def d(self, p, theta, phi):
        #theta = np.pi/2
        #phi = 0
        return 2*(1-p)*np.sin(theta/2)*np.cos(theta/2) # d

    def l(self, p, theta, phi):
        p0 = p
        p1 = (1-p)/2
        p2 = (1-p)/2
        return (2)*(sqrt((p0**2)+(p1**2)+(p2**2)-(p0*p1)-(p0*p2)-(p1*p2)))

    def hw(self, p, theta, phi):
        #theta = np.pi/2
        #phi = 0
        return np.sqrt(5)*((3*p-1)/6) #'hw'
    
    
    
    

    
    

#------------------------------------------------------------------------------------------------------
    def l2(self, p, theta, phi):
        return np.sqrt(1-p)*np.sin(np.pi/2)*np.cos(0/2) #'l'

    def non_markov_t_Bellomo(self, lamb, t):
        gamma_0 = 6
        d = sqrt(2*gamma_0*lamb-lamb**2)
        result = exp(-lamb*t)*(cos(d*t/2)+(lamb/d)*sin(d*t/2))**2
        return result

    def non_markov_t_Ana(self, lamb, t):
        
        result = 1 - exp(-lamb*t)*(cos(t/2)+(lamb)*sin(t/2))
        return result

    def plot_coh(self, map_name, x, y, label, theta, phi):
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')

        mpl.rcParams['text.usetex'] = True
        th = f'{str(theta)[0:4]}'
        fi = f'{str(phi)[0:4]}'
        fancy_name = self.name_changer(map_name)
        psi = fr'$|\psi({th},{fi})\rangle$.'
        m = r"Estado inicial $|\psi(\theta,\phi)\rangle =$ " + psi
        if map_name == 'hw':
            #psi = fr'$\frac(|0\rangle+|1\rangle+|2\rangle\psi({th},{fi})\rangle)$.'
            m = r"Estado inicial $|\psi\rangle = \frac{1}{\sqrt{3}}(|0\rangle+|1\rangle+|2\rangle)$ "
        plt.title(m,usetex=True)
        plt.suptitle(fancy_name)
        if map_name == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p')
        plt.ylabel('coerência')
        plt.plot(x,y,label=label)
        # plt.scatter(x,y,label=label)
        # plt.xscale('log')
        plt.ylabel('coerência')
        plt.xlabel('t')
        plt.title(m)
        plt.legend()
        #plt.show()

    def state_B(self):
        alpha = sqrt(0.5)
        beta = sqrt(0.5)
        state = Matrix([[0, 0, 0, 0],
                    [0, alpha**2 ,alpha*beta , 0],
                    [0, beta*alpha, beta**2, 0],
                    [0, 0, 0, 0]])
        # print(state)
        return state

    def state_B2(self, rho, p):

        rho2 = Matrix([[rho[0,0]*p**2, rho[0,1]*p**(3/2), rho[0,2]*p**(3/2), rho[0,3]*p],
                   [rho[0,1]*p**(3/2), rho[1,1]*p+rho[0,0]*p(1-p), rho[1,2]*p, sqrt(p)*(rho[1,3]+rho[0,2]*(1-p))],
                   [rho[0,2]*p**(3/2), rho[1,2]*p, rho[2,2]*p+rho[0,0]*p(1-p), sqrt(p)*(rho[2,3]+rho[0,1]*(1-p))],
                   [rho[0,3]*p, sqrt(p)*(rho[1,3]+rho[0,2]*(1-p)), sqrt(p)*(rho[2,3]+rho[0,1]*(1-p)), 1-(rho[0,0]+rho[1,1]+rho[2,2])]])
        return rho2

    def plot_pd(self):
        lamb = 0.01
        x1 = np.linspace(0,1,100)
        z = np.linspace(0.01,1000,100)
        x = np.array([self.non_markov_t_Ana(lamb,i) for i in z])
        xb = np.array([self.non_markov_t_Bellomo(lamb,i) for i in z])
        # print(x)
        #x = get_list_p_noMarkov(x,'Ana')
        print(type(x1))
        print(type(x))
        print(x1)
        # print(x)
        #x = [i/max(x) for i in x]
        #x = get_list_p_noMarkov(x,'Bellomo')
        # x = [i/max(x) for i in x]
        th = np.pi/2
        ph = 0
        y1 = self.pd(x1, th, ph)
        y = self.pd(x, th, ph)
        yb = self.pd(xb, th, ph)
        # self.plot_coh('pd', x1, y1, label='Markoviano',theta=th,phi=ph)
        self.plot_coh('pd', z, y, label='não Markoviano_A',theta=th,phi=ph)
        # self.plot_coh('pd', z, yb, label='não Markoviano_B',theta=th,phi=ph)
        #plt.scatter(x1,y);plt.scatter(x,y);plt.xscale('log');plt.title()
        plt.show()

    def plot_ad(self):
        lamb = 0.01
        x1 = np.linspace(0,1,1000)
        z = np.linspace(0.01,300,1000)
        xa = np.array([self.non_markov_t_Ana(lamb,i) for i in z])
        xb = np.array([self.non_markov_t_Bellomo(lamb,i) for i in z])
        # print(x)
        #x = get_list_p_noMarkov(x,'Ana')
        print(type(x1))
        print(type(xa))
        print(x1)
        # print(x)
        #x = [i/max(x) for i in x]
        #x = get_list_p_noMarkov(x,'Bellomo')
        # x = [i/max(x) for i in x]
        th = np.pi/2
        ph = 0
        y1 = self.ad(x1, th, ph)
        ya = self.ad(xa, th, ph)
        yb = self.ad(xb, th, ph)
        self.plot_coh('ad', x1, y1, label='no-Markovianity',theta=th,phi=ph)
        #self.plot_coh('pd', z, ya, label='Markovianity_A',theta=th,phi=ph) #1
        #self.plot_coh('pd', z, yb, label='Markovianity_B',theta=th,phi=ph) #1
        self.plot_coh('pd', xa, ya, label='Markovianity_A',theta=th,phi=ph) #2
        self.plot_coh('pd', xb, yb, label='Markovianity_B',theta=th,phi=ph) #2
        # self.plot_coh('ad', x1, ya, label='Markovianity_A',theta=th,phi=ph) #3
        # self.plot_coh('ad', x1, yb, label='Markovianity_B',theta=th,phi=ph) #3
        #plt.scatter(x1,y);plt.scatter(x,y);plt.xscale('log');plt.title()
        plt.show()

    def plot_all_results(self, map_name):
        lamb = 0.01
        x1 = np.linspace(0,1,300)
        x2 = np.linspace(0.01,100,21)
        xa = np.array([self.non_markov_t_Ana(lamb,i) for i in x2])
        plt.ylabel('coerência')
        self.plot_storaged('bf',True)
        self.plot_storaged('bf',False)
        self.plot_theoric(x1,'bf',theta=pi/2,phi=pi/2,descript='Teórico')
        self.plot_theoric(xa,'bf',theta=pi/2,phi=pi/2,descript='Teórico2')
        plt.xlabel('p (Markov) ; t(n-Markov)')
        # a.plot_theoric_n_Markov(xa,'bf',theta=pi/2,phi=pi/2,descript='bf - n Markov')
        plt.xscale('log')
        # a.plot_theoric(xa,'bf',theta=pi/2,phi=pi/2,descript='bf no Markov')
        plt.legend()
        plt.xlim(0.01)
        plt.show()

from numpy import cos, sin, sqrt, pi, exp


def main():
    from numpy import cos, sin, sqrt, pi, exp
    a = TheoricMaps()
    lamb = 0.01
    x1 = np.linspace(0,1,300)
    x2 = np.linspace(0.01,100,21)
    xa = np.array([a.non_markov_t_Ana(lamb,i) for i in x2])
    plt.ylabel('coerência')
    a.plot_storaged('bf',False)
    a.plot_storaged('bf',True)
    # a.plot_theoric(x1,'bf',theta=pi/2,phi=pi/2,descript='Teórico')
    a.plot_theoric(xa,'bf',theta=pi/2,phi=pi/2,descript='Teórico')
    plt.xlabel('p (Markov) ; p(t) (n-Markov)')
    # a.plot_theoric_n_Markov(xa,'bf',theta=pi/2,phi=pi/2,descript='bf - n Markov')
    plt.xscale('log')
    # a.plot_theoric(xa,'bf',theta=pi/2,phi=pi/2,descript='bf no Markov')
    plt.legend()
    plt.xlim(0.01)
    plt.show()
    sys.exit()
    a.plot_pd()
    a.plot_ad()
    # sys.exit()
    #a.print_state()
    #--------- para plotar os mapas para diferentes valores de theta e phi:-------
    # a.plot_all_theoric_space('ad')
    #a.plot_all_theoric_space('pf')
    #a.plot_all_theoric_space('bf')
    #a.plot_all_theoric_space('bpf')
    #a.plot_all_theoric_space('d')
    #a.plot_all_theoric_space('adg')
    #a.plot_all_theoric_space('l')
    #a.plot_all_theoric_space('hw')
    #-----------------------------------------------------------------------------
    
    #--------- para plotar todos os dados salvos com os valores teóricos:---------
    #x = np.linspace(-100,100,21)
    ###x = [0, pi/4, 3*pi/4, pi]
    lamb = 0.01
    x1 = np.linspace(0,1,10)
    x2 = np.linspace(0.01,1000,10)
    xa = np.array([a.non_markov_t_Ana(lamb,i) for i in x2])
    xb = np.array([a.non_markov_t_Bellomo(lamb,i) for i in x2])
    #x = [i/max(x) for i in x]
    rho = a.state_B()
    print(a.state_B())
    #p = 2
    #rho2 = Matrix([[rho[0,0]*p**2, rho[0,1]*p**(3/2), rho[0,2]*p**(3/2), rho[0,3]*p],
                   #[0,rho[1,1]*p+rho[0,0]*p(1-p),0,0],
                   #[0,0,rho[2,2]*p+rho[0,0]*p(1-p),0],
                   #[0,0,0,1-(rho[0,0]+rho[1,1]+rho[2,2])]])

    #sys.exit()
    print('x1 = ',x1)
    print('x2 = ',x2)
    print('xa = ',xa)
    print('xb = ',xb)
    th = np.pi/2
    ph = 0
    # map_name = 'bpf'
    # y1 = a.map_choser_(map_name)(xa, th, ph)
    y1 = a.bpf(x1, th, ph)
    print(y1)
    ya = a.bpf(x2, th, ph)
    yb = a.bpf(x2, th, ph)
    a.plot_coh('bpf', x1, y1, label='Markovianity',theta=th,phi=ph)
    #plt.show()
    # a.plot_coh('bpf', xa, ya, label='Markovianity_A',theta=th,phi=ph)
    a.plot_coh('bpf', x2, ya, label='No-Markovianity_A',theta=th,phi=ph) #3
    # plt.xscale('log') #2
    # plt.show()
    # a.plot_coh('bpf', xb, yb, label='Markovianity_B',theta=th,phi=ph) #2
    a.plot_coh('bpf', x2, yb, label='No-Markovianity_B',theta=th,phi=ph) #3
    # plt.xscale('log')
    # a.plot_coh('bpf', x2, ya, label='Markovianity_A',theta=th,phi=ph) #3
    # a.plot_coh('bpf', x2, yb, label='Markovianity_B',theta=th,phi=ph) #3
    # a.plot_coh('bpf', x1, ya, label='Markovianity_A',theta=th,phi=ph) #1
    # a.plot_coh('bpf', x1, yb, label='Markovianity_B',theta=th,phi=ph) #1
    plt.show()

    sys.exit()

    # a.plot_storaged('adg',True)
    # a.plot_theoric(x1,'adg',theta=pi/2,phi=0)
    # plt.legend(loc=1)
    # plt.show()
# 
    # a.plot_storaged('ad',True)
    # a.plot_theoric(x1,'ad',theta=pi/2,phi=0)
    # plt.legend(loc=1)
    # plt.show()
# 
    # a.plot_storaged('pf',True)
    # a.plot_theoric(x1,'pf',theta=pi/2,phi=0)
    # plt.legend(loc=1)
    # plt.show()

    #a.plot_storaged('pd',False)
    #a.plot_theoric(x1,'pd',theta=pi/2,phi=0)
    #plt.legend(loc=1)
    #plt.show()
#
    #a.plot_storaged('bf',True)
    #a.plot_theoric(x1,'bf',theta=pi/2,phi=pi/2)
    #plt.legend(loc=1)
    #plt.show()

    a.plot_storaged('bpf',False)
    #a.plot_coh('bpf', xa, ya, label='Markovianity_A',theta=th,phi=ph)
    #a.plot_coh('pd', xb, yb, label='Markovianity_B',theta=th,phi=ph) #2

    a.plot_theoric(x1,'bpf',theta=pi/2,phi=0.0,descript='isometria')
    a.plot_theoric(xa,'bpf',theta=pi/2,phi=0.0,descript='isometria_A')
    a.plot_theoric(xb,'bpf',theta=pi/2,phi=0.0,descript='isometria_B')
    plt.legend(loc=1)
    plt.show()
    s.exit()
    a.plot_storaged('d',True)
    a.plot_theoric(x1,'d',theta=pi/2,phi=0)
    plt.legend(loc=1)
    plt.show()

    a.plot_storaged('l',True)
    a.plot_theoric(x1,'l',theta=pi/2,phi=0)
    plt.legend(loc=1)
    plt.show()

    a.plot_storaged('adg',True)
    a.plot_theoric(x1,'adg',theta=pi/2,phi=0)
    plt.legend(loc=1)
    plt.show()
# 
    a.plot_storaged('hw',True)
    a.plot_theoric(x1,'hw',theta=pi/2,phi=0)
    plt.legend(loc=1)
    plt.show()
    #-----------------------------------------------------------------------------
    #state = a.theoric_rho_A_ad
    #print(print_latex(a.coherence(state)))
if __name__ == "__main__":
    main()
