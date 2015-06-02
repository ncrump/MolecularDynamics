"""
Created on Sun Feb 15 20:47:16 2015
CSI 786, Assignment 2
Nick Crump
"""

# Version 0
# Simple MD Using Verlet Velocity Method

"""
Molecular dynamics simulation of Lennard-Jones particles
in 3D using Verlet-Velocity finite difference method.
"""

import numpy as np
import CubicLattice as lat
from datetime import datetime
import matplotlib.pyplot as plt

t0 = datetime.now()

# function to calculate LJ potential and forces
#----------------------------------------------
def LJ(px,py,pz,fx,fy,fz,nprt):
    fx,fy,fz = fx*0,fy*0,fz*0
    pe = 0
    for i in range(nprt-1):
        for j in range(i+1,nprt):
            dx  = px[i]-px[j]
            dy  = py[i]-py[j]
            dz  = pz[i]-pz[j]
            rij  = (dx**2 + dy**2 + dz**2)**0.5
            fij  = 48*(rij**-13) - 24*(rij**-7)
            pe  += rij**-12 - rij**-6
            dxr = dx/rij
            dyr = dy/rij
            dzr = dz/rij
            fx[i] += fij*dxr
            fy[i] += fij*dyr
            fz[i] += fij*dzr
            fx[j] -= fij*dxr
            fy[j] -= fij*dyr
            fz[j] -= fij*dzr
    return fx,fy,fz,4.0*pe
#----------------------------------------------

# function to calculate kinetic energy
#----------------------------------------------
def kinetic(vx,vy,vz):
    ke  = 0.5*np.sum((vx**2 + vy**2 + vz**2))
    return ke
#----------------------------------------------

# define input parameters
#----------------------------------------------
nprt = 4      # number of particles
temp = 0.1    # temperature
nstp = 10000  # number of steps
dt   = 0.005  # time step
r0   = 1.2    # distance between particles
outfile = 'trajectory_4atms_T0.2_run0.xyz'
#----------------------------------------------

# initialize positions to FCC lattice
pxn,pyn,pzn = lat.FCC(r0,0)

# initialize velocities from normal distribution
sig = temp**0.5
vxn = np.random.normal(0,sig,nprt)
vyn = np.random.normal(0,sig,nprt)
vzn = np.random.normal(0,sig,nprt)

# initialize forces
fxn = np.zeros(nprt)
fyn = np.zeros(nprt)
fzn = np.zeros(nprt)

# calculate initial values
fxn,fyn,fzn,aveP = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt)
aveK = kinetic(vxn,vyn,vzn)
aveE = aveP + aveK
tmpE = 0

# storage arrays
apx,apy,apz,avx,avy,avz         = [pxn],[pyn],[pzn],[vxn],[vyn],[vzn]
aveParr,aveKarr,aveEarr,stdEarr = [aveP],[aveK],[aveE],[tmpE]
t = np.arange(nstp)*dt

# advance time step
step = 1
while step < nstp:
    # update positions
    pxn = pxn + vxn*dt + 0.5*(fxn*dt**2)
    pyn = pyn + vyn*dt + 0.5*(fyn*dt**2)
    pzn = pzn + vzn*dt + 0.5*(fzn*dt**2)

    # update forces
    fxi = np.copy(fxn)
    fyi = np.copy(fyn)
    fzi = np.copy(fzn)
    fxn,fyn,fzn,pe = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt)

    # update velocities
    vxn = vxn + 0.5*(fxi+fxn)*dt
    vyn = vyn + 0.5*(fyi+fyn)*dt
    vzn = vzn + 0.5*(fzi+fzn)*dt

    # calculate energies
    ke = kinetic(vxn,vyn,vzn)
    te = pe + ke

    # update running averages
    aveOldP = aveP
    aveOldK = aveK
    aveOldE = aveE
    aveP = aveP + (pe-aveP)/step
    aveK = aveK + (ke-aveK)/step
    aveE = aveE + (te-aveE)/step
    tmpE = tmpE + (te-aveOldE)*(te-aveE)
    stdE = (tmpE/step)**0.5

    # store and increment step
    apx.append(pxn)
    apy.append(pyn)
    apz.append(pzn)
    avx.append(vxn)
    avy.append(vyn)
    avz.append(vzn)
    aveParr.append(aveP)
    aveKarr.append(aveK)
    aveEarr.append(aveE)
    stdEarr.append(stdE)
    step += 1

# calculate actual temp from kinetic energy
Tact = (2*aveK)/(3*nprt)

# print averages
print '\nT = %1.3f dt = %1.3f <P> = %1.6f <K> = %1.6f <E> = %1.6f stdE = %1.6f' \
      % (temp,dt,aveP,aveK,aveE,stdE)

# plot average potential and kinetic energy
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,aveParr,'g-')
plt.xlim(-1,nstp*dt)
plt.ylabel('$<PE>$',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t,aveKarr,'r-')
plt.xlim(-1,nstp*dt)
plt.ylabel('$<KE>$',fontsize=14)
plt.xlabel('time',fontsize=14)

# plot average and standard dev of total energy
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,aveEarr,'b-')
plt.ticklabel_format(style='sci',useOffset=False,axis='y',scilimits=(-3,3))
plt.xlim(-1,nstp*dt)
plt.ylabel('$<E>$',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t,stdEarr,'b-')
plt.xlim(-1,nstp*dt)
plt.ylabel(r'$\sigma _{E}$',fontsize=16)
plt.xlabel('time',fontsize=14)

# plot histogram of velocities
plt.figure()
plt.subplot(3,1,1)
plt.hist(np.array(avx).flatten())
plt.ylim(0,nstp)
plt.xlim(-1,1)
plt.ylabel('$Vx$',fontsize=14)
plt.subplot(3,1,2)
plt.hist(np.array(avy).flatten())
plt.ylim(0,nstp)
plt.xlim(-1,1)
plt.ylabel('$Vy$',fontsize=14)
plt.subplot(3,1,3)
plt.hist(np.array(avz).flatten())
plt.ylim(0,nstp)
plt.xlim(-1,1)
plt.ylabel('$Vz$',fontsize=14)

# plot average temp
plt.figure()
plt.plot(t,(2*np.array(aveKarr))/(3*nprt),'r-')
plt.xlim(-1,nstp*dt)
plt.ylabel('Temperature',fontsize=14)
plt.xlabel('time',fontsize=14)

# write output trajectory file
f = open(outfile,'w')
for i in range(nstp):
    f.write(str(nprt)+'\n')
    f.write('trajectory at time = '+str(t[i])+'\n')
    for j in range(nprt):
        f.write('%2i %1.6f %1.6f %1.6f\n' %
               (7,apx[i][j],apy[i][j],apz[i][j]))
f.close()

print 'runtime =',datetime.now()-t0