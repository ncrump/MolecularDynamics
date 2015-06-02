"""
Created on Sun Feb 15 20:47:16 2015
CSI 786, Assignment 2
Nick Crump
"""

# MD Using Verlet Velocity Method
"""
Molecular dynamics simulation of Lennard-Jones particles
in 3D using Verlet-Velocity finite difference method.
"""

import numpy as np
import CubicLattice as lat
import matplotlib.pyplot as plt

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
ttot = 5     # total time
temp = 0.1    # temperature
r0   = 1.2    # distance between particles
#----------------------------------------------

# initialize positions to FCC lattice
pxi,pyi,pzi = lat.FCC(r0,0)

# initialize velocities from normal distribution
sig = temp**0.5
vxi = np.random.normal(0,sig,nprt)
vyi = np.random.normal(0,sig,nprt)
vzi = np.random.normal(0,sig,nprt)

# storage arrays
dtarr = []
sdarr = []

for k in [0.05/(2**i) for i in range(8)]:

    dt   = k             # time step
    nstp = int(ttot/dt)  # number of steps

    # reset positions and velocities
    pxn,pyn,pzn = pxi,pyi,pzi
    vxn,vyn,vzn = vxi,vyi,vzi

    # initialize forces
    fxn = np.zeros(nprt)
    fyn = np.zeros(nprt)
    fzn = np.zeros(nprt)

    # calculate initial values
    fxn,fyn,fzn,aveP = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt)
    aveK = kinetic(vxn,vyn,vzn)
    aveE = aveP + aveK
    tmpE = 0

    # advance time step
    step = 1
    time = dt
    while time < ttot:
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

        # increment step
        step += 1
        time += dt

    # store values
    dtarr.append(dt)
    sdarr.append(stdE**0.5)

    # print averages
    print 'T = %1.3f dt = %1.3f <P> = %1.6f <K> = %1.6f <E> = %1.6f stdE = %1.6f' \
          % (temp,dt,aveP,aveK,aveE,stdE)

# make plot
plt.figure()
plt.plot(dtarr,sdarr,'bo-')
plt.xlabel('$dt$',fontsize=16)
plt.ylabel(r'$\sqrt{\sigma _{E}}$',fontsize=16)