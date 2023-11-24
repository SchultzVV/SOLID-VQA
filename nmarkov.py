from numpy import cos, sin, sqrt, pi, exp
import sys 
sys.path.append('src')
from src.theoric_channels import *
a = TheoricMaps()

lamb = 0.01
x1 = np.linspace(0,1,21)
#x = [i/max(x) for i in x]
th = np.pi/2
ph = np.pi/2
#y1 = a.bpf(x1, th, ph)
#ya = a.bpf(xa, th, ph)
#yb = a.bpf(xb, th, ph)


# a.plot_theoric(x1,'ad',theta=pi/2,phi=0,descript='Amplitude Damping')
a.plot_storaged('ad', False)
plt.legend(loc=1)
# plt.show()

# a.plot_theoric(x1,'pf',theta=pi/2,phi=0,descript='Phase-Flip')
# a.plot_storaged('pf', False)
# plt.legend(loc=1)
# plt.show()

# a.plot_theoric(x1,'pd',theta=pi/2,phi=0,descript='Phase-Damping')
a.plot_storaged('pd', False)
plt.legend(loc=1)
# plt.show()

# a.plot_theoric(x1,'bf',theta=pi/2,phi=pi/2,descript='Bit-Flip')
# a.plot_storaged('bf', False)
# plt.legend(loc=1)
# plt.show()

# a.plot_theoric(x1,'bpf',theta=pi/2,phi=0.0,descript='Bit-Phase-Flip')
a.plot_storaged('bpf', False)
# plt.legend(loc=1)
# plt.show()
#s.exit()
# a.plot_theoric(x1,'d',theta=pi/2,phi=0,descript='Depolarizing')
# a.plot_storaged('d', False)
# plt.legend(loc=1)
# plt.show()

# a.plot_theoric(x1,'l',theta=pi/2,phi=0,descript='Lorentz')
a.plot_storaged('l', False)
plt.legend(loc=1)
plt.show()

#plt.show()
# a.plot_theoric(x1,'hw',theta=pi/2,phi=0,descript='H-W dephasing')
# a.plot_storaged('hw', False)
# plt.legend(loc=1)
# plt.show()
# 
# a.plot_theoric(x1,'adg',theta=pi/2,phi=0,descript='Amplitude damping generalizado')
# a.plot_storaged('adg', False)
# plt.legend(loc=1)
# 
# plt.show()