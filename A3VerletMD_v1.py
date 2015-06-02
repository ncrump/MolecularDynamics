"""
Created on Sun Feb 15 20:47:16 2015
CSI 786, Assignment 3
Nick Crump
"""

# Version 1
# MD Using Verlet Velocity Method
# No periodic boundary conditions
# FCC order parameter


"""
Molecular dynamics simulation of Lennard-Jones atoms
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
    # zero forces and potential
    fx,fy,fz = fx*0,fy*0,fz*0
    pe = 0
    # loop through atoms
    for i in range(nprt-1):
        for j in range(i+1,nprt):
            # get pair separation
            dx  = px[i]-px[j]
            dy  = py[i]-py[j]
            dz  = pz[i]-pz[j]
            rij  = (dx**2 + dy**2 + dz**2)**0.5
            fij  = 48*(rij**-13) - 24*(rij**-7)
            pe  += rij**-12 - rij**-6
            # get unit vectors
            dxr = dx/rij
            dyr = dy/rij
            dzr = dz/rij
            # get pair force components
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

# function to calculate FCC order parameter
#----------------------------------------------
def orderFCC(px,py,pz,a,nprt):
    ordX = np.sum(np.cos(4.0*np.pi*px/a))
    ordY = np.sum(np.cos(4.0*np.pi*py/a))
    ordZ = np.sum(np.cos(4.0*np.pi*pz/a))
    ordP = (ordX + ordY + ordZ)/(3.0*nprt)
    return ordP
#----------------------------------------------


# define input parameters
#----------------------------------------------
nstp = 10000     # number of steps
dt   = 0.005     # time step
a    = 1.6       # side length of lattice
k    = 1         # translations of unit cell
#----------------------------------------------

# print number of atoms in lattice
nprt = 4*(k+1)**3
print '\nnumber of atoms = ',nprt

# initialize positions to FCC lattice
pxn,pyn,pzn = lat.FCC(0.707*a,k)
pxn[0],pyn[0],pzn[0] = 0.01,-0.01,0.05

# initialize velocities to zero
vxn = np.zeros(nprt)
vyn = np.zeros(nprt)
vzn = np.zeros(nprt)

# initialize forces to zero
fxn = np.zeros(nprt)
fyn = np.zeros(nprt)
fzn = np.zeros(nprt)

# get initial values
fxn,fyn,fzn,aveP     = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt)
aveK                 = kinetic(vxn,vyn,vzn)
aveE                 = aveP + aveK
aveT                 = (2*aveK)/(3*nprt)
order                = orderFCC(pxn,pyn,pzn,a,nprt)
t                    = np.arange(nstp)*dt
tmpP,tmpK,tmpE,tmpT  = 0,0,0,0

# storage arrays
VXarr,VYarr,VZarr    = [vxn], [vyn], [vzn]
Parr,aveParr,stdParr = [aveP],[aveP],[tmpP]
Karr,aveKarr,stdKarr = [aveK],[aveK],[tmpK]
Earr,aveEarr,stdEarr = [aveE],[aveE],[tmpE]
Tarr,aveTarr,stdTarr = [aveT],[aveT],[tmpT]
ordArr               = [order]

# advance time step: verlet method
step = 1
while step < nstp:
    # update positions
    pxn = pxn + vxn*dt + 0.5*fxn*dt**2
    pyn = pyn + vyn*dt + 0.5*fyn*dt**2
    pzn = pzn + vzn*dt + 0.5*fzn*dt**2

    # update partial velocities from old force
    vxn = vxn + 0.5*fxn*dt
    vyn = vyn + 0.5*fyn*dt
    vzn = vzn + 0.5*fzn*dt

    # update forces
    fxn,fyn,fzn,Pi = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt)

    # update partial velocities from new force
    vxn = vxn + 0.5*fxn*dt
    vyn = vyn + 0.5*fyn*dt
    vzn = vzn + 0.5*fzn*dt

    # get energies and temp
    Ki = kinetic(vxn,vyn,vzn)
    Ei = Pi + Ki
    Ti = (2*Ki)/(3*nprt)

    # get FCC order
    order = orderFCC(pxn,pyn,pzn,a,nprt)

    # update running averages
    aveOldP = aveP
    aveOldK = aveK
    aveOldE = aveE
    aveOldT = aveT
    aveP = aveP + (Pi-aveP)/step
    aveK = aveK + (Ki-aveK)/step
    aveE = aveE + (Ei-aveE)/step
    aveT = aveT + (Ti-aveT)/step
    tmpP = tmpP + (Pi-aveOldP)*(Pi-aveP)
    tmpK = tmpK + (Ki-aveOldK)*(Ki-aveK)
    tmpE = tmpE + (Ei-aveOldE)*(Ei-aveE)
    tmpT = tmpT + (Ti-aveOldT)*(Ti-aveT)
    stdP = (tmpP/step)**0.5
    stdK = (tmpK/step)**0.5
    stdE = (tmpE/step)**0.5
    stdT = (tmpT/step)**0.5

    # store and increment step
    VXarr.append(vxn)
    VYarr.append(vyn)
    VZarr.append(vzn)
    Parr.append(Pi)
    Karr.append(Ki)
    Earr.append(Ei)
    Tarr.append(Ti)
    aveParr.append(aveP)
    aveKarr.append(aveK)
    aveEarr.append(aveE)
    aveTarr.append(aveT)
    stdParr.append(stdP)
    stdKarr.append(stdK)
    stdEarr.append(stdE)
    stdTarr.append(stdT)
    ordArr.append(order)
    step += 1

# print averages
print '<T> = %8.3f sigT = %7.5f' % (aveT,stdT)
print '<P> = %8.3f sigP = %7.5f' % (aveP,stdP)
print '<K> = %8.3f sigK = %7.5f' % (aveK,stdK)
print '<E> = %8.3f sigE = %7.5f' % (aveE,stdE)

## store final positions for Avogadro
#atoms = np.ones(nprt)*7
#stack = np.column_stack((atoms,pxn,pyn,pzn))
#np.savetxt('FinalPos_32atms.xyz',stack,fmt=('%i','%f4','%f4','%f4'))

# plot potential energy
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,Parr,'b-')
plt.plot(t,aveParr,'r-')
plt.ylabel('$<PE>$',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t,stdParr,'r-')
plt.ylabel(r'$\sigma _{PE}$',fontsize=16)
plt.xlabel('time',fontsize=14)

# plot kinetic energy
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,Karr,'b-')
plt.plot(t,aveKarr,'r-')
plt.ylabel('$<KE>$',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t,stdKarr,'r-')
plt.ylabel(r'$\sigma _{KE}$',fontsize=16)
plt.xlabel('time',fontsize=14)

# plot total energy
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,Earr,'b-')
plt.plot(t,aveEarr,'r-')
plt.ticklabel_format(style='sci',useOffset=False,axis='y')
plt.ylabel('$<E>$',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t,stdEarr,'r-')
plt.ylabel(r'$\sigma _{E}$',fontsize=16)
plt.xlabel('time',fontsize=14)

# plot temperature
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,Tarr,'b-')
plt.plot(t,aveTarr,'r-')
plt.ylabel('$<T>$',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t,stdTarr,'r-')
plt.ylabel(r'$\sigma _{T}$',fontsize=16)
plt.xlabel('time',fontsize=14)

# plot histogram of velocities
plt.figure()
plt.subplot(3,1,1)
plt.hist(np.array(VXarr).flatten(),bins=20)
plt.ylabel('$Vx$',fontsize=14)
plt.subplot(3,1,2)
plt.hist(np.array(VYarr).flatten(),bins=20)
plt.ylabel('$Vy$',fontsize=14)
plt.subplot(3,1,3)
plt.hist(np.array(VZarr).flatten(),bins=20)
plt.ylabel('$Vz$',fontsize=14)

# plot histogram of energies
plt.figure()
plt.subplot(2,1,1)
plt.hist(np.array(Parr).flatten(),bins=20)
plt.ylabel('$PE$',fontsize=14)
plt.subplot(2,1,2)
plt.hist(np.array(Karr).flatten(),bins=20)
plt.ylabel('$KE$',fontsize=14)

# plot FCC order parameter
plt.figure()
plt.plot(t,ordArr,'g-')
plt.ylabel('FCC Order')
plt.xlabel('time')

# print runtime
print 'runtime =',datetime.now()-t0