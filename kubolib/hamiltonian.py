#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script hamiltonian.ipynb')


# # Hamiltonian methods

# In[3]:


def bonds_to_matrix(self, bonds, offsets, flag):
    """ Build the Hamiltonian or velocity operator. 
    This functionality is for testing purposes, mainly.
    The vectors are indexed as follows:
    
    (x,y,o) = n = x + y*Lx + o*Lx*Ly
    
    flags: 0 - Ham
           1 - vx
           2 - vy
           3 - vb
    """
    
    Lx = self.Lx
    Ly = self.Ly
    No = self.Norb
    N = self.N
        
    NB = len(bonds)
    matrix = np.zeros([N,N], dtype=complex)
    
    # Run over all the bonds
    for q in range(NB):
        ham = bonds[q]
        ox,oy,b1,b2 = offsets[q]
        
        diff = self.orb_pos[b2] - self.orb_pos[b1] + ox*self.primitives[0] + oy*self.primitives[1]
        dv = -99999
        
        # Choose the kind of operator
        if flag == 0: dv = 1 # Hamiltonian
        elif flag == 1: dv = diff[0]*1j # velocity x
        elif flag == 2: dv = diff[1]*1j # velocity y
        elif flag == 3:
            dist = np.linalg.norm(diff)
            vec_x = np.array([1,0])
            dot = np.vdot(diff,vec_x)
            # print(dist, vec_x, dot)
            sign = dot/np.abs(dot)
            dv = sign*dist*1j
            # print(dv)
        
        
        
        if ox == 0:
            start1x, end1x, start2x, end2x = 0, Lx, 0, Lx
        elif ox > 0:
            start1x, end1x, start2x, end2x = ox, Lx, 0, Lx-ox
        elif ox < 0:
            start1x, end1x, start2x, end2x = 0, Lx+ox, -ox, Lx
        
        if oy == 0:
            start1y, end1y, start2y, end2y = 0, Ly, 0, Ly
        elif oy > 0:
            start1y, end1y, start2y, end2y = oy, Ly, 0, Ly-oy
        elif oy < 0:
            start1y, end1y, start2y, end2y = 0, Ly+oy, -oy, Ly
            
        for i in range(start1x, end1x):
            for j in range(start1y, end1y):
                n =    i +      j*Lx + b2*Lx*Ly
                m = i-ox + (j-oy)*Lx + b1*Lx*Ly
                
                hh = ham[i-start1x,j-start1y]*dv
                matrix[n,m] += hh
        
    return matrix

def get_H(self):
    # Build the Hamiltonian matrix
    if not self.bonds or not self.offsets: print("bonds or offsets not set")
    if len(self.bonds) == len(self.bonds0): print("ramp was not set")
    self.H = bonds_to_matrix(self, self.bonds, self.offsets,0)
    
def get_H0(self):
    if not self.bonds0 or not self.offsets0: print("bonds0 or offsets0 not set")
    self.H0 = bonds_to_matrix(self, self.bonds0, self.offsets0,0)
    
def diag_H(self):
    if isinstance(self.H, int): print("H is not set")
    self.vals, self.vecs = np.linalg.eigh(self.H)
    self.P = self.vecs.transpose().conjugate()
    
def diag_H0(self):
    if isinstance(self.H0, int): print("H0 is not set")
    self.vals0, self.vecs0 = np.linalg.eigh(self.H0)
    self.P0 = self.vecs0.transpose().conjugate()
    
def get_element(self, bonds, offsets, left, right):
    """ Fetch an element from the Hamiltonian operator without having
    to construct it explicitly: <left| H | right>
    
    right and left are provided as a tuple (xx,yy,oo) 
    xx: position along the first primitive vector
    yy: position along the second primitive vector
    oo: orbital
    """
    
    Lx,Ly,No = self.Lx, self.Ly, self.Norb
    
    element = 0*1j
    
    NB = len(offsets) # number of bonds
    
    # Run over all the bonds
    for q in range(NB):
        ham = bonds[q]
        ox,oy,b1,b2 = offsets[q]
        
        cond1 = left[0] - right[0] == ox and left[1] - right[1] == oy
        if cond1 and left[2] == b2 and right[2] == b1:
            
            if ox == 0:
                start1x, end1x, start2x, end2x = 0, Lx, 0, Lx
            elif ox > 0:
                start1x, end1x, start2x, end2x = ox, Lx, 0, Lx-ox
            elif ox < 0:
                start1x, end1x, start2x, end2x = 0, Lx+ox, -ox, Lx

            if oy == 0:
                start1y, end1y, start2y, end2y = 0, Ly, 0, Ly
            elif oy > 0:
                start1y, end1y, start2y, end2y = oy, Ly, 0, Ly-oy
            elif oy < 0:
                start1y, end1y, start2y, end2y = 0, Ly+oy, -oy, Ly

            i = left[0]
            j = left[1]
            element += ham[i-start1x,j-start1y]
        
    return element


# # Hamiltonian KPM methods

# In[4]:



            
def mult_HV_bonds(self, new, temp, hams, offsets, factor,flag):
    """ Efficient implementation of the product by the Hamiltonian 
    prims: list of primitive vectors
    opos: list of the orbitals' positions
    flag = 0 - Hamiltonian
    flag = 1 - velocity x
    flag = 2 - velocity y
    """    
    
    if self.DEBUG:
        print(f"entered mult_HV_bonds {factor=} {flag=}")
    
    Lx,Ly,No = self.Lx, self.Ly, self.Norb
    
    NB = len(offsets)
    
    for i in range(NB):
        ham = hams[i]
        ox,oy,b1,b2 = offsets[i]
        
        # Difference of the atoms' positions
        diff = self.orb_pos[b2] - self.orb_pos[b1] + ox*self.primitives[0] + oy*self.primitives[1]
        
        dv = -99999 # Stupid number so that the program really breaks 
        
        if flag == 0:
            dv = 1
        elif flag == 1:
            dv = diff[0]*1j
        elif flag == 2:
            dv = diff[1]*1j
        elif flag == 3:
            dist = np.linalg.norm(diff)
            vec_x = np.array([1,0])
            dot = np.vdot(diff,vec_x)
            sign = dot/np.abs(dot)
            dv = sign*dist*1j
        
        if self.DEBUG: print(f"bond {i}. {dv=}")
        
        if ox == 0:
            start1x, end1x, start2x, end2x = 0, Lx, 0, Lx
        elif ox > 0:
            start1x, end1x, start2x, end2x = ox, Lx, 0, Lx-ox
        elif ox < 0:
            start1x, end1x, start2x, end2x = 0, Lx+ox, -ox, Lx
        
        if oy == 0:
            start1y, end1y, start2y, end2y = 0, Ly, 0, Ly
        elif oy > 0:
            start1y, end1y, start2y, end2y = oy, Ly, 0, Ly-oy
        elif oy < 0:
            start1y, end1y, start2y, end2y = 0, Ly+oy, -oy, Ly
            
        new[start1x:end1x,start1y:end1y,b2] += dv*factor*ham*temp[start2x:end2x,start2y:end2y,b1]
        
    if self.DEBUG:
        print("left mult_HV_bonds")
            
def mult_H0(self, new, temp, factor):
    mult_HV_bonds(self, new, temp, self.bonds0, self.offsets0, factor,0)
    
def mult_H(self, new, temp, factor):
    mult_HV_bonds(self, new, temp, self.bonds, self.offsets, factor,0)
    
def mult_V(self, new, temp, axis):
    # axis = 0 is vx, axis = 1 is vy
    # use bonds0 so that the ramp isn't even considered
    flag = axis + 1
    mult_HV_bonds(self, new, temp, self.bonds0, self.offsets0, 1,flag)
    
def mult_Vx(self, new, temp):
    mult_V(self, new, temp, 0)
    
def mult_Vy(self, new, temp):
    mult_V(self, new, temp, 1)

def mult_Vb(self, new, temp):
    mult_V(self, new, temp, 2)


# # New Hamiltonian
# Generalized for PBC and time-dependency

# In[5]:



            
def hamiltonian_g(self, write, read, hams, offsets, factors):
    """ Efficient implementation of the product by the Hamiltonian 
    """    
    
    if self.DEBUG:
        print(f"entered hamiltonian_g {factor=} {flag=}")
    
    Lx,Ly,No = self.Lx, self.Ly, self.Norb
    NB = len(offsets)
    if len(factors) != NB: print("hamiltonian_g: number of factors must be equal to number of bonds")
        
    
    for i in range(NB):
        dx,dy,b1,b2 = offsets[i]
        
        if self.DEBUG: print(f"bond {i}")
        
        start1x = max(0, dx)
        start2x = max(0,-dx)
        start1y = max(0, dy)
        start2y = max(0,-dy)
            
        end1x = start1x + Lx - abs(dx) 
        end2x = start2x + Lx - abs(dx) 
        end1y = start1y + Ly - abs(dy) 
        end2y = start2y + Ly - abs(dy)
        # print("bond",i)
        # print(start1x, end1x)
        # print(start2x, end2x)
        # print(start1y, end1y)
        # print(start2y, end2y)
        
        # Bulk multiplication (works for PBC and OBC)
        write[start1x:end1x,start1y:end1y,b2] += factors[i]*hams[i][start2x:end2x,start2y:end2y]*read[start2x:end2x,start2y:end2y,b1]
        
        # PBC         
        if dy == 0 and dx != 0:
            
            a,b = 0,0
            if dx > 0:
                b = Lx-1
            else:
                a = Lx-1
            
        
            h = factors[i]*hams[i][b,:]
            write[a,:,b2] += h*read[b,:,b1]
            
        elif dx == 0 and dy != 0:
            
            a,b = 0,0
            if dy > 0:
                b = Ly-1
            else:
                a = Ly-1
            h = factors[i]*hams[i][:,b]
            write[:,a,b2] += h*read[:,b,b1]
        
        elif dx != 0 and dy != 0:
            # vertical lines 
            a,b,c,ap,bp,cp = 0,0,0,0,0,0
            if dx > 0:
                a = Lx-1
            if dy < 0:
                b = 1
            ap = (a+dx+Lx)%Lx
            bp = 1-b
            c  = b+Ly-1
            cp = bp+Ly-1
            

            h = factors[i]*hams[i][a,b:c]
            write[ap,bp:cp,b2] += h*read[a,b:c,b1]
            
            
            # horizontal lines
            a,b,c,ap,bp,cp = 0,0,0,0,0,0
            if dy > 0:
                a = Ly-1
            if dx < 0:
                b = 1

            ap = (a+dy+Ly)%Ly
            bp = 1-b
            c  = b+Lx-1
            cp = bp+Lx-1
            
            h = factors[i]*hams[i][b:c,a]
            write[bp:cp,ap,b2] += h*read[b:c,a,b1]
            
            
            # corners
            a,b = 0,0
            if dx > 0:
                a = Lx-1
            if dy > 0:
                b = Ly-1
            ap = (a+dx+Lx)%Lx
            bp = (b+dy+Ly)%Ly
                        
            write[ap,bp,b2] += factors[i]*hams[i][a,b]*read[a,b,b1]

            
        
    if self.DEBUG: print("left hamiltonian_g")
 


# In[ ]:




