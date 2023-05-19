import os
import MDAnalysis
from newanalysis.helpers import calcEnergyAtomicVoro
from newanalysis.unfold import minDistCenterBox
from newanalysis.functions import atomsPerResidue, residueFirstAtom
import numpy as np
from copy import deepcopy
from sys import exit
import time
import sys
import h5py
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

maxshell=3

isolute=0
nmol=1001
boxl=round(u1.coord.dimensions[0],4)

times = np.concatenate((np.arange(0.0,0.101,0.001),
                        np.arange(0.11,1.01,0.01),
                        np.arange(1.1,10.1,0.1),
                        np.arange(10.5,50.5,0.5)))

framesets = [(0,360,1)]


tlen = times.shape[0]
epa = np.zeros((tlen,mqs1.n_atoms,maxshell+1))

listf=np.arange(1,51)
folders = [base+"{number}".format(number=i) for i in listf]
henergy=np.zeros((tlen,len(folders),maxshell+1))

nat = mqs1.n_atoms
acoord=np.zeros((3))
ratom=np.zeros((tlen))
relax=np.zeros((tlen))
r=np.zeros((tlen))

ctr3=0

for f in folders:
    print("Entering folder "+f+" ..")
    apr = atomsPerResidue(sel)
    rfa= residueFirstAtom(sel)

    h5file = h5py.File("voro_"+str(ctr3+1)+"/neq_delaunay.hdf5",'r')
    dataset = h5file['delaunay']

    index=str(listf[ctr3])
    u10=MDAnalysis.Universe(base+"mqs1_swm4_1000.psf", base+"crd/1mq-swm4_cold_"+index+".crd")
    sel=u10.select_atoms('all')
    mqs1=u10.select_atoms('resname MQS1')
    coor=np.ascontiguousarray(sel.positions,dtype='double')
    coms=sel.center_of_mass(compound='residues')
    com_mq = mqs1.center_of_mass()
    minDistCenterBox(com_mq, coms, coor, boxl, apr, rfa)
  
    ds=np.copy(dataset[0])
    epa[0] += calcEnergyAtomicVoro(coor,charges,apr,rfa,isolute,nmol,ds,maxshell)
    for i in range(maxshell+1):
        henergy[0,ctr3,i]=calcEnergyAtomicVoro(coor,charges,apr,rfa,isolute,nmol,ds,maxshell)[:,i].sum()

    coms_old=coms

    list=[]
    for i in range(nmol-1):
        if dataset[0,i]==1:
            list.append(i)
    n=np.zeros((tlen,len(list)))
    for j in range(len(list)):
        n[0,j]=1
    relax[0]+=np.mean(n[0,:])
    
    u1=MDAnalysis.Universe(base+"mqs1_swm4_1000.psf",f+"/1mq_swm4_1.dcd",f+"/1mq_swm4_2.dcd",f+"/1mq_swm4_3.dcd",f+"/1mq_swm4_4.dcd")
    u1.trajectory[0]
    sel=u1.select_atoms('all')
    mqs1=u1.select_atoms('resname MQS1')

    comw=np.zeros((3))
    dist=np.zeros(len(list))
    distatom=np.zeros(len(list))
    start = time.time()
    move=np.zeros((len(list),3))
    squared_dist=np.zeros((len(list),nat))

    ctr=1
    
    start = time.time()
#    print("")

    for fs in framesets:
        for ts in u1.trajectory[fs[0]:fs[1]:fs[2]]:
#            print("\033[1AFrame %d of %d (%4.1f%%)" % (ctr,tlen,ctr/tlen*100),
 #                 "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
#                  "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/ctr * (tlen-ctr)))[:7]))
        
            coor=np.ascontiguousarray(sel.positions,dtype='double')
            coms=sel.center_of_mass(compound='residues')
            com_mq = mqs1.center_of_mass()

            minDistCenterBox(com_mq, coms, coor, boxl, apr, rfa)
            ds=np.copy(dataset[ctr])

            epa[ctr] += calcEnergyAtomicVoro(coor,charges,apr,rfa,isolute,nmol,ds,maxshell)

            for i in range(maxshell+1):
                henergy[ctr,ctr3,i]=calcEnergyAtomicVoro(coor,charges,apr,rfa,isolute,nmol,ds,maxshell)[:,i].sum()

            for j in range(len(list)):
                for i in range(3):
                    if (coms[list[j]+1,i]-coms_old[list[j]+1,i]) > boxl/2:
                        move[j,i]-=1
                    if (coms[list[j]+1,i]-coms_old[list[j]+1,i]) < -boxl/2:
                        move[j,i]+=1

                if move[j,0]==0 and move[j,1]==0 and move[j,2]==0 and dataset[ctr,list[j]]==1:
                    n[ctr,j]=1
                comw=coms[list[j]+1]+([move[j,0]*boxl,move[j,1]*boxl, move[j,2]*boxl])
                dist[j]=(comw[0]**2+comw[1]**2+comw[2]**2)**0.5

                for i in range(nat):
                    acoord=coor[i]
                    squared_dist[j,i]=(acoord[0]-comw[0])**2+(acoord[1]-comw[1])**2+(acoord[2]-comw[2])**2
                distatom[j]=(np.amin(squared_dist[j,:]))**0.5
            ratom[ctr]+=np.mean(distatom[:])
            r[ctr]+=np.mean(dist[:])
            relax[ctr]+=np.mean(n[ctr,:])
            coms_old=coms

            ctr+=1
    ctr3+=1

nsys=len(folders)
ratom/=float(nsys)
relax/=float(nsys)
r/=float(nsys)
np.savetxt("distance_inf_atom.dat",np.c_[times,ratom])
np.savetxt("distance_inf.dat",np.c_[times,r])
np.savetxt("residence_time_inf.dat",np.c_[times,relax])

epa/=float(nsys)
np.savetxt("raw_shift_total.dat",np.c_[times,epa.sum(axis=(1,2))])
np.savetxt("raw_shift_per_shell.dat",np.c_[times,epa.sum(axis=(1))])
np.savetxt("raw_shift_per_atom.dat",np.c_[times,epa.sum(axis=(2))])                                      
energy = epa.sum(axis=(1,2))
norm=energy[0]-np.mean(energy[-30:])

eps =  np.zeros((tlen,maxshell+1))

for j in range(tlen):
    for i in range(maxshell+1):
        eps[j,i] = epa[j,:,i].sum()
        
for i in range(maxshell+1):
    eps_avg = np.mean(eps[-30:,i])
    eps[:,i] -= eps_avg

eps[:,:] /= norm


epn=np.zeros((tlen,mqs1.n_atoms))

for j in range(tlen):
    for i in range(mqs1.n_atoms):
        epn[j,i]=epa[j,i,:].sum()
        
for i in range(nat):
    epn_avg = np.mean(epn[-30:,i])
    epn[:,i] -= epn_avg

epn[:,:] /= norm

np.savetxt("shift_per_shell.dat",np.c_[times,eps])
np.savetxt("shift_per_atom.dat",np.c_[times,epn])
np.savetxt("shift_total.dat",np.c_[times,(energy[:]-np.mean(energy[-30:]))/norm])



binmin=0.0
binmax=150.0
bins=150
binwidth=(binmax-binmin)/bins
xenergy=np.arange(binmin,binmax,binwidth)+binwidth/2
    
for j in range(maxshell+1): 
    shell=str(j+1)
    f=open("energy_histo_shell_"+shell+".dat",'w')
    f.close
    print("Calculating histogram for shell "+shell+" ..")
    for i in range(tlen):
        histo=np.histogram(henergy[i,:,j],bins=bins,range=[binmin,binmax],density=True)

        f=open("energy_histo_shell_"+shell+".dat",'ab')
        np.savetxt(f,np.c_[np.full((xenergy.shape[0]),times[i]),xenergy,histo[0]])
        f.close()
        
        f=open("energy_histo_shell_"+shell+".dat",'a')
        f.write("\n")
        f.close()

print("Calculating histogram for all shells ..")
               
energyall=np.zeros((tlen,len(folders)))
            
for j in range(tlen):
    for i in range(len(folders)):
        energyall[j,i]=henergy[j,i,:].sum()

f=open("energy_histo_shell_all.dat",'w')
f.close
for i in range(tlen):
    histo=np.histogram(energyall[i],bins=bins,range=[binmin,binmax],density=True)
    
    f=open("energy_histo_shell_all.dat",'ab')
    np.savetxt(f,np.c_[np.full((xenergy.shape[0]),times[i]),xenergy,histo[0]])
    f.close()
    f=open("energy_histo_shell_all.dat",'a')
    f.write("\n")
    f.close()
