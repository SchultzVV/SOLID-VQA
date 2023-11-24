
import sys
sys.path.append('/home/v/Dissertation')
from src.simulation_with_save import *
from numpy import cos, sin, exp, linspace, meshgrid, array, zeros, shape

def non_markov_t_Bellomo(lamb,t):
    gamma_0 = 2.8
    d = sqrt(2*gamma_0*lamb-lamb**2)
    result = exp(-lamb*t)*(cos(d*t/2)+(lamb/d)*sin(d*t/2))**2
    return result

def non_markov_t_Ana(lamb,t):
    result = 1 - exp(-lamb*t)*(cos(t/2)+lamb*sin(t/2))
    return result

def plot_space(mode):
    size = 200
    t = linspace(0,60,size)
    lamb_list = linspace(0,0.08,size)
    z = zeros((size,size));    j = 0;    k = 0
    for p in t:
        for lamb in lamb_list:
            if mode == 'Ana':
                z[j,k] = non_markov_t_Ana(lamb,p)
            if mode == 'Bellomo':
                z[j,k] = non_markov_t_Bellomo(lamb,p)
            j+=1
        k+=1
        j=0
    t, lamb_list = meshgrid(t,lamb_list)

    print(shape(t))
    print(shape(lamb_list))
    print(shape(z))

    import matplotlib.pyplot as plt
    #import pylab
    #from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(t, lamb_list,z)
    #ax.set_title(f'{mode}')
    ax.set_xlabel('t')
    ax.set_ylabel('lambda')
    ax.set_zlabel('p(t,lambda)')
    plt.show()

plot_space('Ana')
# plot_space('Bellomo')