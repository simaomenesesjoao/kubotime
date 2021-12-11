#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script models.ipynb')


# # Models

# In[3]:


def set_graphene_nanoribbon(self, Lx, Ly, sample, dV, SCALE=3.5, calc=False, PBC=False):
    a0 = 1.0
    primitives = [[3*a0, 0.0], [0.0, np.sqrt(3)*a0]]
    orb_pos = [[    0.0,              0.0 ], 
               [     a0,              0.0 ], 
               [ 3*a0/2,  np.sqrt(3)*a0/2 ], 
               [ 5*a0/2,  np.sqrt(3)*a0/2 ]]

    self.set_prim(primitives)
    self.set_orb_pos(orb_pos)
    self.set_geometry(Lx,Ly,sample)
    self.set_scale(SCALE)
    self.set_ramp(dV, calc=calc)

    t = 1.0
    # Define the TB bonds: within same unit cell
    self.add_bond(t*np.ones([  Lx,   Ly]), [ 0, 0, 0, 1])
    self.add_bond(t*np.ones([  Lx,   Ly]), [ 0, 0, 1, 0])
    self.add_bond(t*np.ones([  Lx,   Ly]), [ 0, 0, 1, 2])
    self.add_bond(t*np.ones([  Lx,   Ly]), [ 0, 0, 2, 1])
    self.add_bond(t*np.ones([  Lx,   Ly]), [ 0, 0, 2, 3])
    self.add_bond(t*np.ones([  Lx,   Ly]), [ 0, 0, 3, 2])

    # Define the TB bonds: to other unit cells
    Lxp = Lx-1
    Lyp = Ly-1
    if PBC:
        Lxp = Lx
        Lyp = Ly
        
    self.add_bond(t*np.ones([Lxp, Ly]), [ 1, 0, 3, 0])
    self.add_bond(t*np.ones([Lxp, Ly]), [-1, 0, 0, 3])

    self.add_bond(t*np.ones([Lx, Lyp]), [ 0, 1, 2, 1])
    self.add_bond(t*np.ones([Lx, Lyp]), [ 0,-1, 1, 2])

    self.add_bond(t*np.ones([Lxp, Lyp]), [ 1, 1, 3, 0])
    self.add_bond(t*np.ones([Lxp, Lyp]), [-1,-1, 0, 3])

    self.add_ramp_as_bonds()

    if calc:
        self.get_H0()
        self.get_H()
        self.get_V()

        # Diagonalize the Hamiltonian matrices
        self.diag_H()
        self.diag_H0()


# In[4]:


def set_1DTB(self, Lx, sample, dV, SCALE=2.1):
    a0 = 1.0
    primitives = [[a0, 0.0], [0.0, a0]]
    orb_pos = [[0.0, 0.0 ]]
    
    Ly = 1

    self.set_prim(primitives)
    self.set_orb_pos(orb_pos)
    self.set_geometry(Lx,Ly,sample)
    self.set_scale(SCALE)
    self.set_ramp(dV, calc=calc)

    t = 1.0

    # Define the TB bonds: to other unit cells
    self.add_bond(t*np.ones([Lx-1,   Ly]), [ 1, 0, 0, 0])
    self.add_bond(t*np.ones([Lx-1,   Ly]), [-1, 0, 0, 0])


    self.add_ramp_as_bonds()

    self.get_H0()
    self.get_H()
    self.get_V()

    # Diagonalize the Hamiltonian matrices
    self.diag_H()
    self.diag_H0()


# In[5]:


def set_square(self, Lx, Ly, sample, dV, SCALE=4.5, calc=False, PBC=False):
    a0 = 1.0
    primitives = [[a0, 0.0], [0.0,a0]]
    orb_pos = [[0.0, 0.0 ]]

    self.PBC = PBC
    self.set_prim(primitives)
    self.set_orb_pos(orb_pos)
    self.set_geometry(Lx,Ly,sample)
    self.set_scale(SCALE)
    self.set_ramp(dV, calc=calc)

    t = 1.0
    
    # Define the TB bonds: to other unit cells
    Lxp = Lx-1
    Lyp = Ly-1
    if PBC:
        Lxp = Lx
        Lyp = Ly
        
    self.add_bond(t*np.ones([Lxp, Ly]), [ 1, 0, 0, 0])
    self.add_bond(t*np.ones([Lxp, Ly]), [-1, 0, 0, 0])
    self.add_bond(t*np.ones([Lx, Lyp]), [ 0, 1, 0, 0])
    self.add_bond(t*np.ones([Lx, Lyp]), [ 0,-1, 0, 0])
    
    
    
    
    pot = np.zeros([Lx, Ly])
    pot = np.random.random([Lx, Ly])*0.0
    # pot[1,1] = 1
    self.add_bond(pot, [0,0,0,0])
    

    self.add_ramp_as_bonds()

    if calc:
        self.get_H0()
        self.get_H()
        self.get_V()

        # Diagonalize the Hamiltonian matrices
        self.diag_H()
        self.diag_H0()


# In[6]:


def set_square_2nd(self, Lx, Ly, sample, dV, SCALE=4.5, calc=False, PBC=False):
    a0 = 1.0
    primitives = [[a0, 0.0], [0.0,a0]]
    orb_pos = [[0.0, 0.0 ]]

    self.set_prim(primitives)
    self.set_orb_pos(orb_pos)
    self.set_geometry(Lx,Ly,sample)
    self.set_scale(SCALE)
    self.set_ramp(dV, calc=calc)

    t = 1.0
    
    # Define the TB bonds: to other unit cells
    Lxp = Lx-1
    Lyp = Ly-1
    if PBC:
        Lxp = Lx
        Lyp = Ly
        
#     self.add_bond(t*np.ones([Lxp, Ly]), [ 1, 0, 0, 0])
#     self.add_bond(t*np.ones([Lxp, Ly]), [-1, 0, 0, 0])
#     self.add_bond(t*np.ones([Lx, Lyp]), [ 0, 1, 0, 0])
#     self.add_bond(t*np.ones([Lx, Lyp]), [ 0,-1, 0, 0])
    
    self.add_bond(t*np.ones([Lxp, Lyp]), [ 1, 1, 0, 0])
    self.add_bond(t*np.ones([Lxp, Lyp]), [-1,-1, 0, 0])
    self.add_bond(t*np.ones([Lxp, Lyp]), [ 1,-1, 0, 0])
    self.add_bond(t*np.ones([Lxp, Lyp]), [-1, 1, 0, 0])

    self.add_ramp_as_bonds()

    if calc:
        self.get_H0()
        self.get_H()
        self.get_V()

        # Diagonalize the Hamiltonian matrices
        self.diag_H()
        self.diag_H0()


# In[ ]:





# In[ ]:




