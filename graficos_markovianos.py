from numpy import cos, sin, sqrt, pi, exp
import sys 
sys.path.append('src')
from src.theoric_channels import *
a = TheoricMaps()

lamb = 0.01
# list_p = np.linspace(0.01,1.6,2001)
# list_p = np.linspace(0.01,2000,2000)# usa esse
list_theoric = np.linspace(0,100,21)# será?
list_p = np.linspace(0,1000,21)# será?
# list_t = np.linspace(0,2000,10)
list_t = np.array([a.non_markov_t_Ana(lamb, i) for i in list_p])
# x2 = np.linspace(50,1000,1000)

#x2 = np.linspace(0, 1000, len(coh_l))

list_of_maps = ['l']
list_of_maps = ['ad','pd','adg','bf','bpf','d','l','hw']
list_of_maps = ['ad','pd','adg','bf','pf','bpf','d','l','hw']
list_of_maps = ['pd']
list_of_maps = ['ad','pd','adg','bf','pf','bpf','d']
list_of_lambs = [0,0.0003,0.003,0.03, 0.3]
list_of_maps = ['ad']
list_of_maps = ['pf','ad']#,'bf','pf']
list_of_lambs = [0,0.0003,0.0009, 0.003, 0.009, 0.03, 0.09, 0.3 ,0.9]
list_of_lambs = [0.01]
th = pi/2
ph = 0
# lambd = 0.01
for map in list_of_maps:
    if map == 'bf':
        ph = pi/2
    else:
        ph = 0
    # a.plot_theoric(list_t,map,theta=th,phi=ph,descript='plot_theoric')
    
    # a.theoric_plot(list_p, map, theta=th, phi=ph, lambd=lamb, descript='', Markovianity=False)
    #a.theoric_plot(list_p, map, theta=th, phi=ph, lambd=1, descript='', Markovianity=False)
    #a.theoric_plot(list_p, map, theta=th, phi=ph, lambd=0.1, descript='', Markovianity=False)
    a.theoric_plot(list_theoric, map, theta=th, phi=ph, lambd=0.01, descript='', Markovianity=False)
    # a.theoric_plot(list_p, map, theta=th, phi=ph, lambd=0, descript='', Markovianity=False)
    # a.plot_theoric_n_Markov(list_t, map, theta=th, phi=ph,lambd=0.01, descript='plot_theoric_n_Markov')

    a.plot_storaged(list_p, map, lamb, False)
    plt.xlabel('t')
    plt.ylabel('coerência')
    plt.xscale('log')
    # plt.xlim(0.01)

    plt.legend()
    plt.show()
sys.exit()
for map in list_of_maps:
    if map == 'bf':
        ph = pi/2
    else:
        ph = 0
    for lambd in list_of_lambs:
        # lambd = 0.1
        a.plot_theoric(list_p,map,theta=pi/2,phi=0,descript='teórico')
        # a.theoric_plot(list_p,map,theta=th,phi=ph,lambd=lambd,descript=' ', Markovianity=False)
        a.plot_storaged(map,list_p,True)
        a.theoric_plot(list_t, map, theta=th, phi=ph, lambd=lambd, descript=' ', Markovianity=True)
        # a.theoric_plot(list_t, map, theta=th, phi=ph,lambd=lambd, descript=' ', Markovianity=False)

        # a.plot_theoric_n_Markov(list_t, map, theta=th, phi=ph,lambd=lambd, descript=' ')
        # lambd = 0.01
        # a.plot_theoric_n_Markov(list_t, map, theta=th, phi=ph,lambd=lambd, descript=' ')
        # lambd = 0.0001
        # a.plot_theoric_n_Markov(list_t, map, theta=th, phi=ph,lambd=lambd, descript=' ')
        # a.plot_theoric_n_Markov_B(x1,map,theta=th,phi=ph,descript='Teórico não Markoviano')
        # a.plot_storaged(map, x2, False)
        if map == 'l':
            plt.xlabel(fr'$\xi$')
        else:
            plt.xlabel('p (Markov) ; t (n-Markov)')
        plt.ylabel('coerência')
        # plt.xlabel('t')

        # plt.xscale('log')
        # plt.xlim(0.01)
        plt.legend()
    plt.show()
sys.exit()    

a.plot_theoric(x1,'ad',theta=pi/2,phi=0,descript='Teórico Markoviano')
a.plot_storaged('ad',True)
a.plot_theoric_n_Markov(x1,'ad',theta=pi/2,phi=0,descript='Teórico não Markoviano')
a.plot_storaged('ad',False)
plt.xlabel('p (Markov) ; t (n-Markov)')
plt.ylabel('coerência')

plt.xscale('log')

plt.legend(loc=1)
plt.show()
sys.exit()
a.plot_theoric(x1,'pd',theta=pi/2,phi=0,descript='Phase-Damping')
a.plot_storaged('pd',False)
plt.legend(loc=1)
plt.show()

# 
a.plot_theoric(x1,'adg',theta=pi/2,phi=0,descript='Amplitude damping generalizado')
a.plot_storaged('adg',False)
plt.legend(loc=1)
plt.show()

a.plot_theoric(x1,'bf',theta=pi/2,phi=pi/2,descript='Bit-Flip')
a.plot_storaged('bf',False)
plt.legend(loc=1)
plt.show()

a.plot_theoric(x1,'pf',theta=pi/2,phi=0,descript='Phase-Flip')
a.plot_storaged('pf',False)
plt.legend(loc=1)
plt.show()

a.plot_theoric(x1,'bpf',theta=pi/2,phi=0.0,descript='Bit-Phase-Flip')
a.plot_storaged('bpf',False)
plt.legend(loc=1)
plt.show()
#s.exit()

a.plot_theoric(x1,'d',theta=pi/2,phi=0,descript='Depolarizing')
a.plot_storaged('d',False)
plt.legend(loc=1)
plt.show()

a.plot_theoric(x1,'l',theta=pi/2,phi=0,descript='Lorentz')
a.plot_storaged('l',False)
plt.legend(loc=1)
plt.show()

#plt.show()
a.plot_theoric(x1,'hw',theta=pi/2,phi=0,descript='H-W dephasing')
a.plot_storaged('hw',False)
plt.legend(loc=1)

plt.show()

