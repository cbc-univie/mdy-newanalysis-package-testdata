import os
import MDAnalysis
from newanalysis.helpers import calcEnergyAtomic
from newanalysis.unfold import minDistCenterBox
from newanalysis.functions import atomsPerResidue, residueFirstAtom
import numpy as np
from copy import deepcopy
from sys import exit
import time
from datetime import timedelta

base='../../data/1mq_swm4_nonequilibrium/'
psf_s0=base+"mqs0_swm4_1000.psf"
psf_s1=base+"mqs1_swm4_1000.psf"
#Check S0 PSF:
if np.array_equal(MDAnalysis.Universe(psf_s0).atoms.masses, MDAnalysis.Universe(psf_s0).atoms.masses.astype(bool)):
    print("Used wrong PSF format for S0 state (masses unreadable!)")
    sys.exit()
#Check S1 PSF:
if np.array_equal(MDAnalysis.Universe(psf_s1).atoms.masses, MDAnalysis.Universe(psf_s1).atoms.masses.astype(bool)):
    print("Used wrong PSF format for S1 state (masses unreadable!)")
    sys.exit()



u0=MDAnalysis.Universe(psf_s0)
u1=MDAnalysis.Universe(psf_s1,base+"1/1mq_swm4_1.dcd",base+"1/1mq_swm4_2.dcd",base+"1/1mq_swm4_3.dcd",base+"1/1mq_swm4_4.dcd")



dq=u1.select_atoms("resname MQS1").charges-u0.select_atoms("resname MQS0").charges
mqs1=u1.select_atoms('resname MQS1')
sel=u1.select_atoms('all')
for i in range(mqs1.n_atoms):
    mqs1.atoms[i].charge=dq[i]


charges=sel.charges
masses=sel.masses

apr = atomsPerResidue(sel)
rfa= residueFirstAtom(sel)

isolute=0
nmol=1001
boxl=round(u1.coord.dimensions[0],4)


times = np.concatenate((np.arange(0.0,0.101,0.001),
                        np.arange(0.11,1.01,0.01),
                        np.arange(1.1,10.1,0.1),
                        np.arange(10.5,50.5,0.5)))

framesets = [(0,360,1)]

tlen = times.shape[0]
epa = np.zeros((tlen,mqs1.n_atoms))


list=np.arange(1,51)
folders = [base+"{number}".format(number=i) for i in list]

henergy=np.zeros((tlen,len(folders)))

ctr3=0

for f in folders:
    print("Entering folder "+f+" ..")
    index=str(list[ctr3])
    u10=MDAnalysis.Universe(psf_s1, base+"crd/1mq-swm4_cold_"+index+".crd")
    sel=u10.select_atoms('all')
    mqs1=u10.select_atoms('resname MQS1')
    coor=np.ascontiguousarray(sel.positions,dtype='double')
    coms=sel.center_of_mass(compound='residues')
    com_mq = mqs1.center_of_mass()
    minDistCenterBox(com_mq, coms, coor, boxl, apr, rfa)
    tmp=calcEnergyAtomic(coor,charges,apr,rfa,isolute,nmol)
    epa[0] +=tmp
    henergy[0,ctr3]=tmp.sum()

    
    u1=MDAnalysis.Universe(psf_s1,f+"/1mq_swm4_1.dcd",f+"/1mq_swm4_2.dcd",f+"/1mq_swm4_3.dcd",f+"/1mq_swm4_4.dcd")
    u1.trajectory[0]

    sel=u1.select_atoms('all')
    mqs1=u1.select_atoms('resname MQS1')

    ctr=1
    
    start = time.time()
#    print("")

    for fs in framesets:
        for ts in u1.trajectory[fs[0]:fs[1]:fs[2]]:
 #           print("\033[1AFrame %d of %d (%4.1f%%)" % (ctr,(tlen-1),ctr/(tlen-1)*100),
 #                 "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
 #                 "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/ctr * (tlen-ctr)))[:7]))
        
            coor=np.ascontiguousarray(sel.positions,dtype='double')
            coms=sel.center_of_mass(compound='residues')
            com_mq =  mqs1.center_of_mass()
            minDistCenterBox(com_mq, coms, coor, boxl, apr, rfa)
            tmp=calcEnergyAtomic(coor,charges,apr,rfa,isolute,nmol)
            epa[ctr] += tmp
            henergy[ctr,ctr3]=tmp.sum()

            ctr+=1
    ctr3+=1



nsys=len(folders)
epa/=float(nsys)


energy = epa.sum(axis=1)
norm=energy[0]-np.mean(energy[-20:])


np.savetxt("shift_absolute.dat",np.c_[times,energy[:]])
np.savetxt("shift_total.dat",np.c_[times,(energy[:]-np.mean(energy[-20:]))/norm])


binmin=-100.0
binmax=200.0
bins=300
binwidth=(binmax-binmin)/bins
xenergy=np.arange(binmin,binmax,binwidth)+binwidth/2
    
        
f=open("energy_histo.dat",'w')
f.close
for i in range(tlen):
    histo=np.histogram(henergy[i],bins=bins,range=[binmin,binmax],density=True)
    f=open("energy_histo.dat",'ab')
    np.savetxt(f,np.c_[np.full((xenergy.shape[0]),times[i]),xenergy,histo[0]])
    f.close()
    f=open("energy_histo.dat",'a')
    f.write("\n")
    f.close()
