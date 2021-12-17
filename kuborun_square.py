import numpy as np
import time
import sys
import os
home=os.path.expanduser('~')
sys.path.insert(1, home+'/projects_sync/codes/kubotime/kubolib')
from kubo_lib import *


def kubosquare_main(Lx,Ly,sample,dV,ander,seed,fermi,NchebT,NchebF, NT, TMAX):
    """ Calculate the kubo formula in time for these parameters. Returns the current with these 
    parameters as well as the current with half the number of Chebyshev polynomials in the
    time evolution operator and the Fermi operator
    """
    norm = sample-1


    kub = kubo()
    kub.set_square(Lx, Ly, sample, dV, calc=False)

    TMIN = 0#; TMAX = 200; NT = 14; 
    tlist = np.linspace(TMIN, TMAX, NT)

    # Add anderson
    anderson = np.zeros([Lx, Ly])
    anderson[kub.lead1:kub.lead2,:] = (np.random.random([sample, Ly]) - 0.5)*ander
    kub.add_bond(anderson, [ 0, 0, 0, 0])

    conv = 3
    NR = 1
    conds = np.zeros([NT, NR, conv])

    flag = 2
    axis = 0
    t1 = time.time()
    np.random.seed(seed); conds[:,:,0] = np.real(kub.kubotime_random_kpm(tlist, fermi, NR, NchebF,    NchebT,    axis, flag))
    t2 = time.time()
    duration = t2-t1
    np.random.seed(seed); conds[:,:,1] = np.real(kub.kubotime_random_kpm(tlist, fermi, NR, NchebF//2, NchebT,    axis, flag))
    np.random.seed(seed); conds[:,:,2] = np.real(kub.kubotime_random_kpm(tlist, fermi, NR, NchebF,    NchebT//2, axis, flag))

    return tlist,conds[:,0,:]/norm, duration


if __name__ == "__main__":
    Lx     = int(sys.argv[1])
    Ly     = int(sys.argv[2])
    sample = int(sys.argv[3])
    dV     = float(sys.argv[4])
    ander  = float(sys.argv[5])
    seed   = int(sys.argv[6])
    fermi  = float(sys.argv[7])
    NchebT = int(sys.argv[8])
    NchebF = int(sys.argv[9])
    NT     = int(sys.argv[10])
    TMAX   = float(sys.argv[11])
    print(f"Lx={Lx}, Ly={Ly}, sample={sample}, dV={dV}, ander={ander}, seed={seed}, fermi={fermi}, NchebT={NchebT}, NchebF={NchebF}, NT={NT}, TMAX={TMAX}")
    tlist,conds,durat = kubosquare_main(Lx,Ly,sample,dV,ander,seed,fermi,NchebT,NchebF, NT, TMAX)
    print(f"{durat:2.2f} seconds")
    rows = len(conds)
    cols = len(conds[0])
    store = np.zeros([rows, cols+1])
    store[:,1:] = conds
    store[:,0] = tlist
    np.savetxt("data.dat", tlist)
