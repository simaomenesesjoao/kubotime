#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


def pot_ramp(dims, sample):
    # Electric field term in 2D. Min is always -0.5 and max
    # is always +0.5
    Lx, Ly, No = dims
    
    lead1 = Lx//2 - sample//2
    lead2 = Lx//2 + sample//2
    
    potential2 = np.zeros(dims)
    
    for i in range(lead1, lead2):
        potential2[i,:,:] = (i - lead1)/sample - 0.5
    for i in range(lead2,Lx):
        potential2[i,:,:] = 0.5
    for i in range(lead1):
        potential2[i,:,:] = -0.5
        
    return potential2

def linearize_pot(pot):
    Lx, Ly, No = pot.shape
    N = Lx*Ly*No
    potential_lin = np.zeros([N,N])
    
    for i in range(Lx):
        for j in range(Ly):
            for o in range(No):
                n = i + j*Lx + o*Lx*Ly
                potential_lin[n,n] = pot[i,j,o]
                
    return potential_lin

