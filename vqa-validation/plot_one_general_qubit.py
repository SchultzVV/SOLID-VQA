from torch.autograd import Variable
import torch
from rsvg import rsvg
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from tools import *
import sys
import pickle
import os

files =['Fidelidades_nqb-1.pkl', 'Fidelidades_nqb-2.pkl', 'Fidelidades_nqb-3.pkl', 'Fidelidades_nqb-4.pkl']
files = ['bibliograph/'+i for i in files]

with open(files[0], 'rb') as f:
    y1 = pickle.load(f)
with open(files[1], 'rb') as f:
    y2 = pickle.load(f)
with open(files[2], 'rb') as f:
    y3 = pickle.load(f)
with open(files[3], 'rb') as f:
    y4 = pickle.load(f)

size = 100
x = np.linspace(1, size, size)
plt.plot(x, y1[0:size], ylabel='Fidelidade')#, linestyle="dashdot")
#plt.plot(x, y2[0:size], label='2 Qubit', linestyle=":")
#plt.plot(x, y3[0:size], label='3 Qubit', linestyle="--")
#plt.plot(x, y4[0:size], label='4 Qubit')
plt.title('Fidelidade entre o estado alvo e preparados pelo VQA')
plt.ylim(0.95, 1.01)
plt.legend()
plt.show()