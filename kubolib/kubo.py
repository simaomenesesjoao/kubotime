#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from aux import *
from velocity import *


# In[2]:


if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script kubo.ipynb')


# # Kubo methods
# 
# $$ \sum_{n}\left\langle n\right|PJPe^{-iHt}f\left(H_{0}\right)e^{iHt}\left|n\right\rangle  $$
# 
# What algorithm should I use to evaluate the current at the points in the time series of NT points? NS = number of points inside the whole sample<br>
# 1. Calculate NT operators, one for each time<br>
# Total time=NS\*(N² + N)\*NT + NT\*N³\*4
# 
# 2. Act on the vectors instead<br>
# Total time=NS\*(3N² + N)\*NT

# # <font size=7><b>Matrix methods</b></font>
# 
# These methods are intended for testing purposes. The full matrices (Hamiltonian matrix, Fermi matrix, time evolution matrix) are calculated explicitly.
# 
# # <b>Matrix methods:</b> Average current
# 
# 

# In[3]:


def kubotime_vector(self, tlist, mu, start, axis):
    """ time-resolved Kubo formula for a given chemical potential
    and starting vector, for a list of times"""
    
    NT = len(tlist)
    tlist2 = tlist*self.SCALE
    dt = tlist2[1] - tlist2[0]
    muR = mu/self.SCALE
    
    lin = linearize(self, start) # linearize
    
    # use the first placeholder matrix to store the 
    # Fermi function, but don't recalculate it 
    if isinstance(self.F, int): 
        self.F = fermi(self, muR)
        norm = np.linalg.norm(self.F)
        herm = np.linalg.norm(self.F - self.F.transpose().conjugate())
        print(f"calculating fermi. Norm={norm} herm={herm}")
    else:
        print("Using precalculated Fermi matrix from previous calculation")
        
    if isinstance(self.U, int): 
        self.U = evolve(self, dt)
        print(f"calculating time evolution dt. Norm={np.linalg.norm(self.U)}")
    else:
        print("Using precalculated time evolution operator from previous calculation")
        
    
    project(self, lin)
    if axis == 0:
        left = self.Vx@lin*self.SCALE    
    elif axis == 2:
        left = self.Vb@lin*self.SCALE    
        
    right = lin*1.0
    project(self, left)
    
    
    conds = np.zeros(NT, dtype=complex)
    for i in range(NT):
        conds[i] = left.conjugate().transpose()@self.F@right
        left = self.U@left
        right = self.U@right
        
    return -conds*np.pi*2

def kubotime_sample(self, tlist, mu, axis):
    """ calculate the kubo formula """ 
    NT = len(tlist)
    conds = np.zeros(NT, dtype=complex)
    
    # run over the sample
    for xx in range(self.lead1, self.lead2):
        # print(xx, end=" ")
        for yy in range(self.Ly):
            for oo in range(self.Norb):
                vector = np.zeros([self.Lx, self.Ly, self.Norb], dtype=complex)
                vector[xx,yy,oo] = 1
                cond = kubotime_vector(self, tlist, mu, vector, axis)
                conds += cond
                
                
    return conds

def kubotime_random(self, tlist, mu, NR, axis, flag):
    """ calculate the kubo formula using random vectors""" 
    NT = len(tlist)
    conds = np.zeros([NT,NR], dtype=complex)
    
    # several random vectors
    for r in range(NR):
        uni = np.random.random([self.Lx, self.Ly, self.Norb]) + 0*1j
        if flag == 0:
            rand = np.exp(uni*np.pi*2j)        
        elif flag == 1:
            rand = np.cos(uni*np.pi*2)*np.sqrt(2)
        elif flag == 2:
            rand = (uni - 0.5)*np.sqrt(12)
            
        rand[:self.lead1,:,:] = 0
        rand[self.lead2:,:,:] = 0
            
        
        cond = kubotime_vector(self, tlist, mu, rand, axis)
        conds[:,r] = cond
                
                
    return conds
    


# # <b>Matrix methods:</b> Local current
# 
# Time-resolved local current calculated across *one* bond
# 
# $$ \text{Tr}\left[J_{ij}e^{-iHt}f\left(H_{0}\right)e^{iHt}\right] $$
# 
# The local current operator is defined between two sites $i$ and $j$. I'm using the designation "site" in a broader sense, since it also encompasses orbital degrees of freedom
# 
# $$ \sum_{n=i,j}\left\langle n\right|J_{ij}e^{-iHt}f\left(H_{0}\right)e^{iHt}\left|n\right\rangle  $$
# 
# Since $ J_{ij} $ is local in real space, it heavily restricts the trace, so that the sum only needs to be performed at $n=i$ and $n=j$

# In[2]:


def kubo_bond(self, tlist, mu, site_i, site_j, axis):
    """ Computes the local current across a bond, from site_i to site_j, 
    with explicit matrix computation. The result is in units of e²/h
    """
    
    tlistR = tlist*self.SCALE
    NT = len(tlist)
    # axis = 0
    dt = tlistR[1] - tlistR[0]
    muR = mu/self.SCALE
    
    # Try to recycle the fermi operator
    if isinstance(self.F, int): 
        self.F = fermi(self, muR)
        norm = np.linalg.norm(self.F)
        herm = np.linalg.norm(self.F - self.F.transpose().conjugate())
        print(f"calculating fermi. Norm={norm} herm={herm}")
    else:
        print("Using precalculated Fermi matrix from previous calculation")
        
    if isinstance(self.U, int): 
        self.U = evolve(self, dt)
        print(f"calculating time evolution dt. Norm={np.linalg.norm(self.U)}")
    else:
        print("Using precalculated time evolution operator from previous calculation")
        
   
    
    # Fetch the local current operator matrix element
    J_ij = get_v_element(self, site_i, site_j,axis)*self.SCALE
    
    sites = [site_i, site_j]
    cond = np.zeros(NT, dtype=complex)
    for site in sites:
        e_n = np.zeros([self.Lx, self.Ly, self.Norb], dtype=complex)
        e_n[site[0], site[1], site[2]] = 1
        
        # sites where v_ij will act
        rr1 = e_n[site_i[0],site_i[1],site_i[2]]
        rr2 = e_n[site_j[0],site_j[1],site_j[2]]

        # Product by the local current operator
        left = np.zeros([self.Lx, self.Ly, self.Norb], dtype=complex)
        left[site_i[0],site_i[1],site_i[2]] = J_ij*rr2
        left[site_j[0],site_j[1],site_j[2]] = np.conj(J_ij)*rr1
        
    
        left_lin = linearize(self, left)
        right_lin = linearize(self, e_n)
        
        for i in range(NT):            
            after = self.F@right_lin
            
            c = left_lin.conjugate().transpose()@after
            cond[i] += c
            right_lin = self.U@right_lin
            left_lin = self.U@left_lin
            
    # print(sites,J_ij,cond[-1]*2*np.pi)
    return cond*2*np.pi


# # <font size=7><b> KPM methods </b></font>
# 
# # <b>KPM methods:</b> KPM auxiliary methods
# 
# KPM expansion of the Fermi function. The method <code>kpm_fermi(self, write, read, mu, Ncheb)</code> uses the non-perturbed Hamiltonian <code>H_0</code> stored in the object in bond form to evaluate the Fermi operator at a given chemical potential $\mu$. 
# 
# $$ f\left(\mu\right)=\frac{1}{1+e^{\beta\left(H_{0}-\mu\right)}}=\sum_{n=0}f_{n}\left(\mu\right)T_{n}\left(H_{0}\right) $$
# 
# KPM expansion of the time evolution operator using the full Hamiltonian <code>H</code> stored in the object in bond form. The method <code> kpm_time(self, write, read, dt, Ncheb) </code> acts on a vector 'read' with the time evolution operator of time increment <code>dt</code> and stores it in the vector 'write'
# 
# $$ e^{iHt/\hbar}=\sum_{n=0}U_{n}\left(t\right)T_{n}\left(H\right) $$
# 
# The method <code>kpm_time_inplace(self, vector, dt, Ncheb)</code> does the same thing but replaces the vector with the time-evolved vector by means of a temporary vector.

# In[5]:


def kpm_fermi(self, write, read, mu, Ncheb):
    # Calculates the action of the Fermi operator on a vector 'read' and
    # stores it in the vector 'write'. The Fermi operator is calculated via a 
    # Chebyshev expansion of the Heaviside step function (zero temperature). The
    # Hamiltonian inside this operator is the unperturbed Hamiltonian H_0
    # Parameters:
    # mu: fermi energy
    # Ncheb: number of Chebyshev polynomials
    
    
    old = read*1.0
    new = read*0.0
    self.mult_H0(new, old, 1)

    write += old*coef_fermi_a(0,mu)*jackson(0,Ncheb+1) 
    write += new*coef_fermi_a(1,mu)*jackson(1,Ncheb+1)
    for k in range(2,Ncheb):
        temp = new*1.0
        new = -old*1.0
        self.mult_H0(new, temp, 2)
        old = temp*1.0

        write += new*coef_fermi_a(k,mu)*jackson(k,Ncheb+1) 

        
def kpm_time(self, write, read, dt, Ncheb):
    # Use the full Hamiltonian (WITH ramp)
    
    old = read*1.0
    new = read*0.0
    self.mult_H(new, old, 1)

    write += old*coef_evol(0,dt) + new*coef_evol(1,dt)
    for k in range(2,Ncheb):
        temp = new*1.0
        new = -old*1.0
        self.mult_H(new, temp, 2)
        old = temp*1.0

        write += new*coef_evol(k,dt)

def kpm_time_inplace(self, vector, dt, Ncheb):
    # evolves 'vector'
    tempv = vector*0.0
    kpm_time(self, tempv, vector, dt, Ncheb)
    vector[:,:,:] = tempv*1.0 # ASSIGNMENT NAO FAZ NADA
    
    
    


# # <b>KPM methods:</b> Average current
# 
# <code>kubotime_vector_kpm(self, tlist, mu, start, NchebF, NchebT, axis)</code> calculates one term of the full trace of the current operator (<i>not</i> the local current operator). This method is the numerical implementation of the formula
# 
# $$ \left\langle n\right|PJP e^{-iHt}f\left(H_{0}\right)e^{iHt}\left|n\right\rangle  $$
# 
# where $|n>$ is the <code>start</code> vector in the code<br>
# 
# The method <code>kubotime_random_kpm(self, tlist, mu, NR, NchebF, NchebT, axis, flag)</code> calls the previous method for a random vector defined only inside the sample. The <code>flag</code> parameter controls the type of random vector being fed into the previous method. <code>axis</code> controls the kind of velocity operator (x direction, y direction, or parallel to hopping direction)
# 

# In[4]:


def kubotime_vector_kpm(self, tlist, mu, start, NchebF, NchebT, axis):
    """ time-resolved Kubo formula for a given chemical potential
    and starting vector, for a list of times, with KPM"""
    
    NT = len(tlist)
    tlist2 = tlist*self.SCALE
    dt = tlist2[1] - tlist2[0]
    muR = mu/self.SCALE
    
    # project starting vector into sample
    project_nonlin(self, start)
    right = start*1.0
    
    # Product by the velocity and project back into sample
    left = start*0.0
    self.mult_V(left, start, axis)
    project_nonlin(self, left)
    
    conds = np.zeros(NT, dtype=complex)
    for i in range(NT):
        temp = start*0.0
        kpm_fermi(self,temp, right, muR, NchebF)
        
        conds[i] = np.sum(left.conjugate()*temp)
        
        # Evolve the vectors in time
        kpm_time_inplace(self,  left, dt, NchebT)
        kpm_time_inplace(self, right, dt, NchebT)
        
    return -conds*self.SCALE*np.pi*2




def kubotime_random_kpm(self, tlist, mu, NR, NchebF, NchebT, axis, flag):
    """ calculate the kubo formula using random vectors
    flag controls the initial random vector:
    0 is uniform in the unit complex circle
    1 is the real part of 1 (MUCH smaller error bar)
    2 is uniform with std_dev 1
    """ 
    
    NT = len(tlist)
    conds = np.zeros([NT,NR], dtype=complex)
    
    # several random vectors
    for r in range(NR):
        
        uni = np.random.random([self.Lx, self.Ly, self.Norb]) + 0*1j
        if flag == 0:
            rand = np.exp(uni*np.pi*2j)        
        elif flag == 1:
            rand = np.cos(uni*np.pi*2)*np.sqrt(2)
        elif flag == 2:
            rand = (uni - 0.5)*np.sqrt(12)
            
        rand[:self.lead1,:,:] = 0
        rand[self.lead2:,:,:] = 0
        
        cond = kubotime_vector_kpm(self, tlist, mu, rand, NchebF, NchebT, axis)
        conds[:,r] = cond
                
                
    return conds
    
    


# # KPM v2
# doesn't work

# In[5]:




def kubotime_vector_kpm_v2(self, tlist, mu, start, NchebF, NchebT, axis):
    """ time-resolved Kubo formula for a given chemical potential
    and starting vector, for a list of times, with KPM
    
    This should be faster AND have less fluctuations.
    It doesn't
    """
    
    NT = len(tlist)
    tlist2 = tlist*self.SCALE
    dt = tlist2[1] - tlist2[0]
    muR = mu/self.SCALE
    
    left = start*1.0
    
    # multiply by Fermi operator
    right = start*0.0
    kpm_fermi(self,right, start, muR, NchebF)
    # right = start*1.0
   
    conds = np.zeros(NT, dtype=complex)
    for i in range(NT):
        
         # Product by the projected velocity
        projected = right*1.0
        project_nonlin(self, projected)
        rightV = start*0.0
        self.mult_V(rightV, projected, axis)
        project_nonlin(self, rightV)
        
        # rightV = projected*1.0
        
        conds[i] = np.sum(left.conjugate()*rightV)
        
        # Evolve the vectors in time
        kpm_time_inplace(self, left, dt, NchebT)
        kpm_time_inplace(self, right, dt, NchebT)
        
        
    return -conds*self.SCALE*np.pi*2



def kubotime_random_kpm_v2(self, tlist, mu, NR, NchebF, NchebT, axis, flag):
    """ calculate the kubo formula using random vectors - version 2""" 
    NT = len(tlist)
    conds = np.zeros([NT,NR], dtype=complex)
    
    # several random vectors
    for r in range(NR):
        
        uni = np.random.random([self.Lx, self.Ly, self.Norb]) + 0*1j
        if flag == 0:
            rand = np.exp(uni*np.pi*2j)        
        elif flag == 1:
            rand = np.cos(uni*np.pi*2)*np.sqrt(2)
        elif flag == 2:
            rand = (uni - 0.5)*np.sqrt(12)
        
        # uni = uni*0
        # uni[self.Lx//2, self.Ly//2, self.Norb//2] = 1
        
        cond = kubotime_vector_kpm_v2(self, tlist, mu, rand, NchebF, NchebT, axis)
        conds[:,r] = cond
                
                
    return conds


# # <font size=7><b>Velocity gauge</b></font>
# 
# # Implementation using the more modern Hamiltonian implementation

# In[8]:


def kpm_time_g(self, write, read, dt, Ncheb, factors):
    # Use the full Hamiltonian (WITH ramp)
    factors2 = [2*fac for fac in factors]
    
    old = read*1.0
    new = read*0.0
    self.hamiltonian_g(new, old, self.bonds, self.offsets, factors)

    write += old*coef_evol(0,dt) + new*coef_evol(1,dt)
    for k in range(2,Ncheb):
        temp = new*1.0
        new = -old*1.0
        self.hamiltonian_g(new, temp, self.bonds, self.offsets, factors2)
        old = temp*1.0

        write += new*coef_evol(k,dt)
        
        

def kpm_time_inplace_g(self, vector, dt, Ncheb, factors):
    # evolves 'vector'
    tempv = vector*0.0
    kpm_time_g(self, tempv, vector, dt, Ncheb, factors)
    vector[:,:,:] = tempv*1.0 # ASSIGNMENT NAO FAZ NADA
    
    
    
        
def kpm_fermi_g(self, write, read, mu, Ncheb):
    # Use the Hamiltonian without the ramp
    NB = len(self.bonds)
    factors = [1 for i in range(NB)]
    factors2 = [2*fac for fac in factors]    
    
    old = read*1.0
    new = read*0.0
    # self.mult_H0(new, old, 1)
    self.hamiltonian_g(new, old, self.bonds, self.offsets, factors)

    write += old*coef_fermi_a(0,mu)*jackson(0,Ncheb+1) 
    write += new*coef_fermi_a(1,mu)*jackson(1,Ncheb+1)
    for k in range(2,Ncheb):
        temp = new*1.0
        new = -old*1.0
        self.hamiltonian_g(new, temp, self.bonds, self.offsets, factors2)
        old = temp*1.0

        write += new*coef_fermi_a(k,mu)*jackson(k,Ncheb+1) 


# In[9]:


def velocity_vector(self, tlist, mu, start, NchebF, NchebT, axis):
    """ average expectation value of the velocity operator (?) """
    
    NT = len(tlist)
    tlist2 = tlist*self.SCALE
    dt = tlist2[1] - tlist2[0]
    muR = mu/self.SCALE
    
    
    
    right = start*1.0
    left = start*0.0
    
    kpm_fermi_g(self,left, start, muR, NchebF)
    
    E = self.E
    conds = np.zeros(NT, dtype=complex)
    for i in range(NT):
        t = tlist2[i]
        factorsU = factor_velgauge_t_exact(self, [E, t, dt])
        factorsV = factor_velocity(self, [axis])*factor_velgauge_t(self, [E, t])
        
        temp = start*0.0
        self.hamiltonian_g(temp, right, self.bonds, self.offsets, factorsV)
        
        
        conds[i] = np.sum(left.conjugate()*temp)
        
        # Evolve the vectors in time
        kpm_time_inplace_g(self, left, dt, NchebT, factorsU)
        kpm_time_inplace_g(self, right, dt, NchebT, factorsU)
        
    return -conds*self.SCALE*np.pi*2


# In[ ]:




