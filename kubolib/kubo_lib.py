#!/usr/bin/env python
# coding: utf-8

# In[1]:


# related files
from aux import *
from ramp import *
from hamiltonian import *
from velocity import *
from kubo import *
from models import *
from tests import *

import numpy as np


# # To do:
# 1. More efficient calculation of KPM fermi and KPM time using ring
# 2. Memory alignment for the Hamiltonian and 
# 
# # Keep in mind
# Pass a numpy array to a function, and then assign it:<br>
# f(a)<br>
# a=b\*1.0<br>
# THIS will do NOTHING because then a becomes a local variable inside the function because of the assignment. modifications are fine. assignments are NOT
# 
# 

# In[2]:


# to convert to script run
if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script *.ipynb')


# In[3]:


class kubo:
    def __init__(self):
        self.DEBUG = False
        
        self.dim = -1
        self.PBC = 293.3
        self.Norb = -1
        self.Lx = -1
        self.Ly = -1
        self.N = -1
        self.sample = -1
        self.SCALE = -1
        self.NB0 = -1
        self.NB = -1
    
        # Coordinates of the leads
        self.lead1 = -1
        self.lead2 = -1

        # Lattice properties
        self.primitives = []
        self.orb_pos = []

        # Potential ramp
        self.dV = ""
        self.ramp = -1
        self.ramp_lin = -1

        # Hamiltonian without ramp
        self.bonds0 = []
        self.offsets0 = []

        self.bonds = []
        self.offsets = []

        # Explicit hamiltonian and velocity matrix (mainly for debugging)
        self.H0 = -1
        self.H  = -1
        self.Vx = -1
        self.Vb = -1

        # Diagonalization
        self.vals0 = -1
        self.vecs0 = -1
        self.P0    = -1 # vec0⁻¹
        self.vals  = -1
        self.vecs  = -1
        self.P     = -1 # vec⁻¹
        

        # Placeholder matrices for more efficient testing
        self.F    = -1 # fermi operator
        self.U    = -1 # time evolution operator
        self.UI   = -1 # inverse time evolution operator
        self.proj = -1 # projection into the sample
        self.D    = -1 # anything
        self.E    = -1
    
    
    # Set the primitive vectors
    def set_prim(self, prims):
        # each line is a primitive vector
        # a1[0] a1[1]
        # a2[0] a2[1] 
        
        self.dim = len(prims)
        self.primitives = np.array(prims)
            
    def set_orb_pos(self, opos):
        # each line is an orbital position
        self.orb_pos = np.array(opos)
        self.Norb = len(opos)
        
    def set_geometry(self, lx, ly, sam):
        if self.Norb == -1: print("Norb is not set")
        self.Lx = lx
        self.Ly = ly
        self.sample = sam
        self.N = lx*ly*self.Norb
        
        # Ramp positions
        self.lead1 = self.Lx//2 - self.sample//2
        self.lead2 = self.lead1 + self.sample
        
        
    def set_ramp(self, dV, calc):
        if self.SCALE == -1: print("SCALE is not set")
        if self.Lx == -1: print("ERROR: geometry not set")
            
        self.dV = dV/self.SCALE
        self.ramp = pot_ramp([self.Lx, self.Ly, self.Norb], self.sample)*self.dV
        if calc:
            self.ramp_lin = linearize_pot(self.ramp)
        
        
    def set_scale(self, scal):
        self.SCALE = scal
        
    def add_bond(self, bond, offset):
        # add the hoppings and rescale them right away
        
        if self.SCALE == -1:
            print("SCALE is not set")
        
        # bonds without ramp
        self.bonds0.append(bond/self.SCALE)
        self.offsets0.append(offset)
        self.NB0 = len(self.bonds0)
        
        # bonds with ramp
        self.bonds.append(bond/self.SCALE)
        self.offsets.append(offset)
        self.NB = len(self.bonds)
        
    def add_ramp_as_bonds(self):
        if not self.bonds0:
            print("bonds are not set,") 
        if not self.offsets0:
            print("offsets are not set")
        if isinstance(self.ramp, int):
            print("ramp is not set")
        
        # self.bonds = self.bonds0.copy()
        # self.offsets = self.offsets0.copy()
        for oo in range(self.Norb):
            self.bonds.append(self.ramp[:,:,oo])
            self.offsets.append([0,0,oo,oo])
        # self.NB = len(self.bonds)
    
    def calculate(self):
        self.get_H0()
        self.get_H()
        self.get_V()

        # Diagonalize the Hamiltonian matrices
        self.diag_H()
        self.diag_H0()
        


# # Bind methods into the Kubo class

# In[4]:


# Hamiltonian methods
kubo.get_matrix_bonds = get_matrix_bonds
kubo.hamiltonian_g = hamiltonian_g
kubo.get_H   = get_H
kubo.get_H0  = get_H0
kubo.diag_H  = diag_H
kubo.diag_H0 = diag_H0

# Get the velocity methods
kubo.get_V = get_V

# Kubo methods
kubo.kubo_bond = kubo_bond
kubo.kubotime_sample = kubotime_sample
kubo.kubotime_vector = kubotime_vector
kubo.kubotime_random = kubotime_random

# Kubo KPM methods
kubo.mult_H0 = mult_H0
kubo.mult_H = mult_H
kubo.mult_V = mult_V
kubo.mult_Vx = mult_Vx
kubo.mult_Vy = mult_Vy
kubo.kubotime_vector_kpm = kubotime_vector_kpm
kubo.kubotime_random_kpm = kubotime_random_kpm
kubo.kubotime_vector_kpm_v2 = kubotime_vector_kpm_v2
kubo.kubotime_random_kpm_v2 = kubotime_random_kpm_v2

# Models (functions defined in models.py)
kubo.set_graphene_nanoribbon = set_graphene_nanoribbon
kubo.set_square = set_square
kubo.set_square_2nd = set_square_2nd
kubo.set_1DTB = set_1DTB

# Tests
kubo.test_tevop = test_tevop
kubo.test_fermi_op = test_fermi_op


# In[ ]:




