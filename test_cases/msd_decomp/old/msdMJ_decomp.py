from __future__ import print_function
import MDAnalysis
import numpy as np
from MDAnalysis.newanalysis.diffusion import msdCOM, msdMJ, unfoldTraj
from diffusion import msdMJdecomp #I just did not have vronis diffusion in my MDAnalysis, thus I used an extra module
from MDAnalysis.newanalysis.correl import correlate
import os
import time
import sys

########################################################################################################################################
def correlXYZ(data, result, ltc=0):
    data = np.ascontiguousarray(data.T)
    for i in range(3):
        tmp = correlate(data[i], data[i], ltc=ltc)
        for j in range(len(data[i])):
            result[j] += tmp[j]

           
########################################################################################################################################
# trajectories
########################################################################################################################################
skip = 20
u=MDAnalysis.Universe("../../data/emim_dca_equilibrium/emim_dca.psf","../../data/emim_dca_equilibrium/emim_dca.dcd")


boxl=round(u.coord.dimensions[0],4)
dt=round(u.trajectory.dt,4)
n = u.trajectory.numframes/skip
if u.trajectory.numframes%skip != 0:
    n+=1


########################################################################################################################################
# molecular species
########################################################################################################################################
sel_cat  = u.selectAtoms("resname EMIM")
sel_an  = u.selectAtoms("resname DCA")


ncat = sel_cat.numberOfResidues()
nan = sel_an.numberOfResidues()


com_cat = np.zeros((int(n),int(ncat),3),dtype=np.float64)
com_an  = np.zeros((int(n),int(nan),3),dtype=np.float64)

mass_cat = sel_cat.masses()
mass_an = sel_an.masses()


charges_cat = sel_cat.charges()
charges_an = sel_an.charges()


########################################################################################################################################
# analysis
########################################################################################################################################
ctr=0
start=time.time()
print("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d" % (ts.frame,u.trajectory.numframes), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    coor_cat = sel_cat.get_positions()
    coor_an = sel_an.get_positions()
    
    com_cat[ctr] = sel_cat.centerOfMassByResidue(coor=coor_cat,masses=mass_cat)
    com_an[ctr] = sel_an.centerOfMassByResidue(coor=coor_an,masses=mass_an)
    ctr+=1

print("unfolding coordinates ..")
unfoldTraj(com_cat,boxl)
unfoldTraj(com_an, boxl)

# Classical mean-squared displacement of coordinates
print("calculating < Delta r^2(t) > ...")
msdcat = msdCOM(com_cat)
filename = 'msd_emim.dat'
f1=open(filename,'w')
for i in range(len(msdcat)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdcat[i]))
f1.close()

msdan = msdCOM(com_an)
filename = 'msd_dca.dat'
f1=open(filename,'w')
for i in range(len(msdan)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdan[i]))
f1.close()


# mean-squared displacement of collective translational dipole moment
print("calculating msdMJ ..")
msdmj = msdMJ(com_cat,com_an)
filename = 'msdMJ.dat'
f1=open(filename,'w')
for i in range(len(msdmj)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj[i]))
f1.close()

# decomposition of mean-squared displacement of collective translational dipole moment
msdmj_cat = msdMJdecomp(com_cat, com_an, com_cat, 1)
filename = 'msdMJ_cat.dat'
f1=open(filename,'w')
for i in range(len(msdmj_cat)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj_cat[i]))
f1.close()

msdmj_an = msdMJdecomp(com_cat, com_an, com_an, -1)
filename = 'msdMJ_an.dat'
f1=open(filename,'w')
for i in range(len(msdmj_an)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj_an[i]))
f1.close()

