import numpy as np
import matplotlib.pyplot as plt
from kubo_lib2 import *
import time

Lx = 16
Ly = 3
sample = 2
dV = 0.0

kub = kubo()
PBC = False
kub.set_graphene_nanoribbon(Lx, Ly, sample, dV, calc=False, PBC=True, SCALE=1)

psi = np.zeros([Lx, Ly, 4])
# psi[Lx//2, Ly//2, 2] = 1
psi += 1
psi1 = psi*0.0
factors = [1 for i in range(len(kub.bonds))]

# Testar se PBC estão bem
kub.hamiltonian_g(psi1, psi, kub.bonds, kub.offsets, factors)
# print(psi)
print(psi1)
