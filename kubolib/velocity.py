#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from hamiltonian import *


# In[2]:


if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script velocity.ipynb')


# # Velocity methods

# In[3]:


def get_v_element(self, left, right, axis):
    """ Calculates the velocity matrix element requested: <left | v | right > 
    without having to construct the whole velocity operator explicitly. 
    The velocity operator does not depend on the ramp, so the Hamiltonian used for
    this calculation can be chosen at once to be the one without ramp
    """
    
    Lx,Ly,No = self.Lx, self.Ly, self.Norb
    n =  left[0] +  left[1]*Lx +  left[2]*Lx*Ly
    m = right[0] + right[1]*Lx + right[2]*Lx*Ly
    a = get_element(self, self.bonds0, self.offsets0, left, right)
    
    vecL =  left[0]*self.primitives[0] +  left[1]*self.primitives[1] + self.orb_pos[ left[2]]
    vecR = right[0]*self.primitives[0] + right[1]*self.primitives[1] + self.orb_pos[right[2]]
    diff = vecL - vecR
    vel = 0
    if axis == 0 or axis == 1:
        vel = -1j*diff[axis]*a
    elif axis == 2:
        dist = np.linalg.norm(diff)
        vec_x = np.array([1.0, 0.0])
        dot = np.vdot(vec_x, diff)
        sign = dot/np.abs(dot)
        vel = -1j*dist*sign*a
        # print("get v_element: ", dist, vel)
    
    
    return vel

def local_v(self, left, right, axis):
    """ Local current operator between two sites/orbitals (right and left)
    This is matrix defined in the whole Hilbert space.
    This matrix is essentially zero except for two places.
    """
    Lx,Ly,No,N = self.Lx, self.Ly, self.Norb, self.N
    
    velocity = np.zeros([N,N], dtype=complex)
    n =   left[0] + left[1]*Lx +  left[2]*Lx*Ly
    m = right[0] + right[1]*Lx + right[2]*Lx*Ly
    vv = get_v_element(self, left, right, axis)
    velocity[m,n] = vv
    velocity[n,m] = np.conj(vv)
    return velocity

def projected_v(self, vels, axis):
    """ sum of local current operators given by the bonds specified in 'vels' """
    Lx,Ly,No,N = self.Lx, self.Ly, self.Norb, self.N
    velocity = np.zeros([N,N], dtype=complex)
    
    NB = len(vels[:,0])
    for i in range(NB):
        
        left = vels[i,:3]
        right = vels[i,3:]
        n =  left[0] +  left[1]*Lx +  left[2]*Lx*Ly
        m = right[0] + right[1]*Lx + right[2]*Lx*Ly
        vv = get_v_element(self, left, right,axis)
        
        velocity[m,n] = vv
        velocity[n,m] = np.conj(vv)
    return velocity

def get_V(self):
    if not self.bonds0 or not self.offsets0: print("bonds0 or offsets0 not set")
    
    # Axis is 0 or 1, flag is 0,1,2 but 0 corresponds to Hamiltonian
    flag = 1
    self.Vx = bonds_to_matrix(self, self.bonds0, self.offsets0,flag)
    print(f"Calculating Vx operator. Norm={np.linalg.norm(self.Vx)}")
    
    flag = 3
    self.Vb = bonds_to_matrix(self, self.bonds0, self.offsets0,flag)
    print(f"Calculating Vb operator. Norm={np.linalg.norm(self.Vb)}")
    

