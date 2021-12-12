import sys
import numpy as np
import kubo_lib as kubo

Lx = int(sys.argv[1]) # Total system length
Ly = int(sys.argv[2]) # width
sample = int(sys.argv[3]) # sample length
SCALE = float(sys.argv[4]) 

# Fermi function
mu = float(sys.argv[5])
Ncheb = int(sys.argv[6]) # number of cheb pols for the Fermi function

# Anderson disorder
W = float(sys.argv[7])
seed = int(sys.argv[8])

# information about the time discretization
TMAX = int(sys.argv[9])
NT = int(sys.argv[10])
NTcheb = int(sys.argv[11])
yprobe = int(sys.argv[12])

# print all the information
# print(f"{Lx=} {Ly=} {sample=} {SCALE=} {mu=} {Ncheb=} {W=} {seed=} {TMAX=} {NT=} {NTcheb=}")
print("Lx",Lx)
print("Ly",Ly)
print("sample", sample)
print("SCALE", SCALE)
print("mu",mu)
print("Ncheb", Ncheb)
print("W",W)
print("seed",seed)
print("TMAX",TMAX)
print("NT",NT)
print("NTcheb",NTcheb)
print("yprobe",yprobe)


# global parameters to be used by any method
No = 4
N = Lx*Ly*No; dims = [Lx, Ly, No]
lead1 = Lx//2 - sample//2
lead2 = lead1 + sample



# Time integration parameters. Assuming time goes from 0 to TMAX in NT steps
# These will be the steps used for integration
tlist = np.linspace(0,TMAX, NT)
dt = tlist[1] - tlist[0]

# Lattice properties
a0 = 1.0
primitives = np.array([[3*a0, 0.0], [0.0, np.sqrt(3)*a0]])
orb_pos = np.array([[0.0, 0.0], [a0, 0.0], [3*a0/2, np.sqrt(3)*a0/2], [5*a0/2, np.sqrt(3)*a0/2]])

# flags
mult_vel = True
mult_t1 = True
mult_t2 = True
mult_fermi = True
mult_ramp = True
flags = [mult_vel, mult_t1, mult_t2, mult_fermi, mult_ramp]

# Chebyshev parameters

conv = 3
kubos = np.zeros([NT, conv])



# Define the Anderson disorder
np.random.seed(seed)
anderson1 = np.zeros([Lx, Ly])
anderson2 = np.zeros([Lx, Ly])
anderson3 = np.zeros([Lx, Ly])
anderson4 = np.zeros([Lx, Ly])
anderson1[lead1:lead2,:] = (np.random.random([sample,Ly]) - 0.5)*W
anderson2[lead1:lead2,:] = (np.random.random([sample,Ly]) - 0.5)*W
anderson3[lead1:lead2,:] = (np.random.random([sample,Ly]) - 0.5)*W
anderson4[lead1:lead2,:] = (np.random.random([sample,Ly]) - 0.5)*W

# Onsite offsets
offA1 = [0,0,0,0] # offset in x direction, offset in y direction, orbital (from), orbital (to)
offA2 = [0,0,1,1]
offA3 = [0,0,2,2]
offA4 = [0,0,3,3]

# Define the TB bonds: within same unit cell
t = 1.0
b1  = t*np.ones([  Lx,   Ly]); off1  = [ 0, 0, 0, 1]
b2  = t*np.ones([  Lx,   Ly]); off2  = [ 0, 0, 1, 0]
b3  = t*np.ones([  Lx,   Ly]); off3  = [ 0, 0, 1, 2]
b4  = t*np.ones([  Lx,   Ly]); off4  = [ 0, 0, 2, 1]
b5  = t*np.ones([  Lx,   Ly]); off5  = [ 0, 0, 2, 3]
b6  = t*np.ones([  Lx,   Ly]); off6  = [ 0, 0, 3, 2]

# Define the TB bonds: to other unit cells
b7  = t*np.ones([Lx-1,   Ly]); off7  = [ 1, 0, 3, 0]
b8  = t*np.ones([Lx-1,   Ly]); off8  = [-1, 0, 0, 3]

b9  = t*np.ones([  Lx, Ly-1]); off9  = [ 0, 1, 2, 1]
b10 = t*np.ones([  Lx, Ly-1]); off10 = [ 0,-1, 1, 2]

b11 = t*np.ones([Lx-1, Ly-1]); off11 = [ 1, 1, 3, 0]
b12 = t*np.ones([Lx-1, Ly-1]); off12 = [-1,-1, 0, 3]




hams = [anderson1, anderson2, anderson3, anderson4, 
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]
hamsR = [ham/SCALE for ham in hams]
offsets = [offA1, offA2, offA3, offA4, 
           off1, off2, off3, off4, off5, off6, off7, off8, off9, off10, off11, off12]

packR = [offsets, hamsR]


potential2 = kubo.potential2o(dims, sample)

# Local current across a horizontal bond inside the same unit cell
px,py = Lx//2, yprobe
vel1 = [px, py, 0]
vel2 = [px, py, 1]

result0 = np.zeros(NT)
result1 = np.zeros(NT)
result2 = np.zeros(NT)
for pos in [vel1, vel2]:
    # print(f"{pos=}",end=" ")
    print("pos",pos)
    n,m,o = pos
    result0 += kubo.KPM2o(SCALE,pos,mu,packR,tlist, potential2, Ncheb, NTcheb, vel1, vel2, primitives, orb_pos, flags)
    result1 += kubo.KPM2o(SCALE,pos,mu,packR,tlist, potential2, Ncheb//2, NTcheb, vel1, vel2, primitives, orb_pos, flags)
    result2 += kubo.KPM2o(SCALE,pos,mu,packR,tlist, potential2, Ncheb, NTcheb//2, vel1, vel2, primitives, orb_pos, flags)
print("")

# Integration 
for i in range(1,NT):
    kubos[i,0] = kubos[i-1,0] + (result0[i-1] + result0[i])/2*dt
    kubos[i,1] = kubos[i-1,1] + (result1[i-1] + result1[i])/2*dt
    kubos[i,2] = kubos[i-1,2] + (result2[i-1] + result2[i])/2*dt

for cc in range(conv):
    np.savetxt(f"kubotime_graphene_nanoribbon_Ly{Ly}_Lx{Lx}_sample{sample}_W{W}_conv{cc}_mu{mu:2.2f}.dat",kubos[:,cc])
