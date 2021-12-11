#!/usr/bin/env python
# coding: utf-8

# # Tests

# In[ ]:


def test_tevop(self):
    dt = 0.1
    U = evolve(self,dt)
    
    total_sum = 0
    
    for i in range(self.N):
        
        vec = self.vecs[:,i]
        val = self.vals[i]



        evolved1 = U@vec
        evolved2 = np.exp(-1j*val*dt)*vec
        dif = np.linalg.norm(evolved1 - evolved2)
    
        total_sum += dif
        
    print(f"Testing time evolution. {total_sum}")
    


def test_fermi_op(self):
    
    total_sum = 0
    mus = [-3, -2, -1, 0, 1, 2, 3]
    for mu in mus:
        
        F = fermi(self, mu)
        trace = np.real(np.sum(np.diag(F)))
        soma = np.sum([1 for i in self.vals0 if i < mu])
        dif = abs(soma - trace)
        total_sum += dif
        
    print("Testing the Fermi operator", total_sum)
    
    
    

