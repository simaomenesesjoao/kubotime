#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import jv


# In[2]:


if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script aux.ipynb')


# # Auxiliary functions

# In[3]:


# Zero temperature Fermi function
def fermi0(e,mu):
    return float(e < mu)

# Fermi function
def fermi(e,b,mu):
    return 1.0/(np.exp(b*(e-mu)) + 1)

# Testing spectral. Expanding Fermi function
def cheb(n,x):
    return np.cos(n*np.arccos(x))

def jackson(n,N):
    arg = np.pi/(N+1)
    term1 = (N-n+1)*np.cos(n*arg)
    term2 = np.sin(n*arg)/np.tan(arg)
    return (term1 + term2)/(N+1)

def coef_fermi_b(n,mu):
    num = 1
    
    if n == 0:
        num = (mu + 1)/2
    elif n == 1:
        num = mu**2/2 - 1/2
    if n > 1:
        num = 0.5*(cheb(n+1,mu)/(n+1) - cheb(n-1,mu)/(n-1)-2*(-1)**n/(n*n-1))
        
    return num/np.pi*2

def coef_fermi_a(n,mu):
    num = 0
    angle = np.arccos(mu)
    if n == 0:
        num = 1-angle/np.pi
    else:
        num = 2/np.pi/n*(np.sin(n*np.pi) - np.sin(n*angle))
        
    return num


def coef_evol(n,t):
    factor = 2
    if n==0:
        factor=1
    return ((-1j)**n)*jv(n,t)*factor


# # More auxiliary functions
# 1. Fermi matrix construction<br>
# 2. Time evolution matrix construction<br>
# 3. Function to project the vector into the sample<br>
# 4. Projector matrix into the sample<br>
# 5. Function which converts (x,y,o) coordinates into a single n coordinate

# In[4]:


def fermi(self, mu):
    # Fermi function
    N = self.N
    fermimat = np.zeros([N,N], dtype=complex)
    for i in range(N):
        E = self.vals0[i]
        if E < mu:
            fermimat[i,i] = 1
    return self.vecs0@fermimat@self.P0

def evolve(self, t):
    """ Using the FULL Hamiltonian, WITH RAMP """
    N = self.N
    tievop = np.zeros([N,N], dtype=complex)
    for i in range(N):
        E = self.vals[i]
        tievop[i,i] = np.exp(-1j*t*E)
    return self.vecs@tievop@self.P

def project(self, vec):
    for i in range(self.Lx):
        if i < self.lead1 or i >= self.lead2:
            for j in range(self.Ly):
                for o in range(self.Norb):
                    n = i + self.Lx*j + self.Lx*self.Ly*o
                    vec[n] = 0
                    
def project_nonlin(self, vec):
    for i in range(self.Lx):
        if i < self.lead1 or i >= self.lead2:
            for j in range(self.Ly):
                for o in range(self.Norb):
                    vec[i,j,o] = 0            
                    
                    
def projector(self):
    """ Matrix which projects into the sample """
    N = self.N
    proj = np.zeros([N,N], dtype=complex)
    for xx in range(self.lead1, self.lead2):
        for yy in range(self.Ly):
            for oo in range(self.Norb):
                n = xx + self.Lx*yy + self.Lx*self.Ly*oo
                proj[n,n] = 1
    return proj
    
                    
def linearize(self, vec):
    lin = np.zeros(self.N, dtype=complex)
    for i in range(self.Lx):
        for j in range(self.Ly):
            for o in range(self.Norb):
                n = i + self.Lx*j + self.Lx*self.Ly*o
                lin[n] = vec[i,j,o]
    return lin

def de_linearize(self, vec):
    delin = np.zeros([self.Lx, self.Ly, self.Norb], dtype=complex)
    for i in range(self.Lx):
        for j in range(self.Ly):
            for o in range(self.Norb):
                n = i + self.Lx*j + self.Lx*self.Ly*o
                delin[i,j,o] = vec[n]
    return delin


# # Even more aux

# In[5]:



# Time evolution assuming the Hamiltonian is constant along each time chunk dt
def factor_velgauge_t(self, args):
    E = args[0]
    t = args[1]
    
    NB = len(self.offsets)
    component = np.zeros(NB)
    for i in range(NB):
        offset = self.offsets[i]
        
        o2 = offset[2]
        o1 = offset[3]
        dr  = offset[0]*self.primitives[0]
        dr += offset[1]*self.primitives[1]
        dr += self.orb_pos[o2] - self.orb_pos[o1]
        
        component[i] = dr@E
    
    return np.exp(1j*t*component)

# Time evolution assuming the Hamiltonian commutes across different times
def factor_velgauge_t_exact(self, args):
    E  = args[0]
    t  = args[1]
    dt = args[2]
    
    NB = len(self.offsets)
    factors = np.zeros(NB, dtype=complex)
    for i in range(NB):
        offset = self.offsets[i]
        
        o2 = offset[2]
        o1 = offset[3]
        dr  = offset[0]*self.primitives[0]
        dr += offset[1]*self.primitives[1]
        dr += self.orb_pos[o2] - self.orb_pos[o1]
        
        delta = dr@E
        if np.abs(delta)<1e-10:
            factors[i] = 1
        else:
            integrated = (np.exp(1j*dt*delta)-1)/(1j*dt*delta)
            factors[i] = np.exp(delta*1j*t)*integrated
        
    
    return factors

def factor_velocity(self, args):
    axis = args[0]
    
    NB = len(self.offsets)
    factors = np.zeros(NB, dtype=complex)
    for i in range(NB):
        offset = self.offsets[i]
        
        o2 = offset[2]
        o1 = offset[3]
        dr  = offset[0]*self.primitives[0]
        dr += offset[1]*self.primitives[1]
        dr += self.orb_pos[o2] - self.orb_pos[o1]
        
        factors[i] = dr[axis]*1j
    
    return factors

def get_matrix_bonds(self):
    mat_bonds = [np.zeros([self.N,self.N], dtype=complex) for k in range(self.NB)]
    # indexation is x + Lx*y + Lx*Ly*o
    Lx = self.Lx
    Ly = self.Ly
    
    for k in range(self.NB):        
        dx,dy,o1,o2 = self.offsets[k]

        for x in range(Lx):        
            for y in range(Ly):
                n =  x           + Lx*y              + Lx*Ly*o1
                m = (x+dx+Lx)%Lx + Lx*((y+dy+Ly)%Ly) + Lx*Ly*o2
                mat_bonds[k][n,m] += self.bonds[k][x,y]
           
    self.mat_bonds = mat_bonds

def get_hamiltonian_matrix(self, args):
    # check if matrix bonds is defined
    E = args[0]
    t = args[1]
    Lx = self.Lx
    Ly = self.Ly
    
    P = projector(self)
    
    H1 = np.zeros([self.N, self.N], dtype=complex)
    H2 = np.zeros([self.N, self.N], dtype=complex)
    factors = factor_velgauge_t(self, [E, t])
    for k in range(self.NB):
        H1 += self.mat_bonds[k]*factors[k]
        H2 += self.mat_bonds[k]
      
    return H2 + P@(H1-H2)@P
        
def get_integrated_hamiltonian_matrix(self, args):
    # check if matrix bonds is defined
    E = args[0]
    t1 = args[1]
    t2 = args[2]
    dt = t2-t1
    Lx = self.Lx
    Ly = self.Ly
    
    P = projector(self)
    
    H1 = np.zeros([self.N, self.N], dtype=complex)
    H2 = np.zeros([self.N, self.N], dtype=complex)
    factors = factor_velgauge_t_exact(self, [E, t1, dt])
    for k in range(self.NB):
        H1 += self.mat_bonds[k]*factors[k]
        H2 += self.mat_bonds[k]
       
    return (H2 + P@(H1-H2)@P)*dt
    


# In[6]:



def get_velocity_matrix(self, args):
    # check if matrix bonds is defined
    axis = args[0] 
    E = args[1]
    t = args[2]
    Lx = self.Lx
    Ly = self.Ly
    
    P = projector(self)
    
    V1 = np.zeros([self.N, self.N], dtype=complex)
    V2 = np.zeros([self.N, self.N], dtype=complex)
    factors1 = factor_velocity(self, [axis])
    factors2 = factor_velgauge_t(self, [E, t])
    for k in range(self.NB):
        V1 += self.mat_bonds[k]*factors1[k]*factors2[k]
        V2 += self.mat_bonds[k]*factors1[k]
        
        
    return V2 + P@(V1-V2)@P

def get_proj_velocity_matrix(self, args):
    
    P = projector(self)
        
    return P@get_velocity_matrix(self, args)@P

def get_local_velocity_matrix(self, args):
    # Local current operator
    P = np.zeros([self.N, self.N], dtype=complex)
    x = self.Lx//2
    y = self.Ly//2
    o = 0
    
    n = x + self.Lx*y + self.Lx*self.Ly*o
    m = (x+1)%self.Lx + self.Lx*y + self.Lx*self.Ly*o
                
    P[n,n] = 1
    P[m,m] = 1
        
    return P@get_velocity_matrix(self, args)@P

def get_local_velocity_matrix_const(self, args):
    axis = args[0] 
    E = args[1]
    t = args[2]
    
    # Local current operator, constant
    
    P = np.zeros([self.N, self.N], dtype=complex)
    x,y,o = self.Lx//2, self.Ly//2, 0
    
    n = x + self.Lx*y + self.Lx*self.Ly*o
    m = (x+1)%self.Lx + self.Lx*y + self.Lx*self.Ly*o
                
    P[n,n] = 1
    P[m,m] = 1
    
    
    V = np.zeros([self.N, self.N], dtype=complex)
    factors1 = factor_velocity(self, [axis])
    # factors2 = factor_velgauge_t(self, [E, t])
    for k in range(self.NB):
        V += self.mat_bonds[k]*factors1[k]
        
    return P@V@P


def get_velocity2(self, args):
    # without oscillatory part
    
    axis = args[0] 
    E = args[1]
    t = args[2]
    Lx = self.Lx
    Ly = self.Ly
    
    P = projector(self)
    
    V = np.zeros([self.N, self.N], dtype=complex)
    factors1 = factor_velocity(self, [axis])
    # factors2 = factor_velgauge_t(self, [E, t])
    for k in range(self.NB):
        V += self.mat_bonds[k]*factors1[k]
        
        
    return P@V@P


# In[7]:


sdfdsfs = 2

