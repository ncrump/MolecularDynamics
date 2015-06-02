"""
Created on Sun Feb 15 20:47:16 2015
CSI 786, Assignment 3
Nick Crump
"""

# Version 2 Cython
# MD using Verlet-velocity method
# No periodic boundary conditions
# FCC order parameter

"""
Molecular dynamics simulation of Lennard-Jones atoms
"""

import cython
import numpy as np
cimport numpy as np
import FCC_Cython as lat
import matplotlib.pyplot as plt
from datetime import datetime

cpdef SimRun():

    t0 = datetime.now()

    # declare C-type variables
    cdef int    nprt,nstp,k,step
    cdef double dt,a,Pi,Ki,Ei,Ti,order
    cdef double aveP,aveK,aveE,aveT,aveOldP,aveOldK,aveOldE,aveOldT
    cdef double tmpP,tmpK,tmpE,tmpT,stdP,stdK,stdE,stdT

    # declare C-type arrays
    cdef np.ndarray[double,ndim=1] pxn,pyn,pzn,vxn,vyn,vzn,fxn,fyn,fzn

    # define input parameters
    #----------------------------------------------
    nstp = 30000     # number of steps
    dt   = 0.005     # time step
    a    = 1.6       # side length of lattice
    k    = 1         # translations of unit cell
    #----------------------------------------------

    # print number of atoms in lattice
    nprt = 4*(k+1)**3
    print '\nnumber of atoms = ',nprt

    # initialize positions to FCC lattice
    pxn,pyn,pzn = lat.FCC(0.707*a,k)

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

    # advance time step: Verlet-velocity method
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
    print '\n<T> = %8.3f sigT = %7.5f' % (aveT,stdT)
    print '<P> = %8.3f sigP = %7.5f' % (aveP,stdP)
    print '<K> = %8.3f sigK = %7.5f' % (aveK,stdK)
    print '<E> = %8.3f sigE = %7.5f\n' % (aveE,stdE)

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
    plt.ylabel('$<E>$',fontsize=14)
    plt.ticklabel_format(style='sci',useOffset=False,axis='y')
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


# function to calculate LJ potential and forces
# applies minimum image convention for PBC
# applies cut and shift of potential
#----------------------------------------------
cpdef LJ(np.ndarray[double] pxn, np.ndarray[double] pyn, np.ndarray[double] pzn,\
         np.ndarray[double] fxn, np.ndarray[double] fyn, np.ndarray[double] fzn,\
         int nprt):
    # declare C-type variables
    cdef int    i,j
    cdef double pe,dx,dy,dz,rij,fij,dxr,dyr,dzr
    # zero forces and potential
    fxn,fyn,fzn = fxn*0,fyn*0,fzn*0
    pe = 0
    # loop through atoms
    for i in range(nprt-1):
        for j in range(i+1,nprt):
            # get pair separation
            dx  = pxn[i]-pxn[j]
            dy  = pyn[i]-pyn[j]
            dz  = pzn[i]-pzn[j]
            rij  = (dx**2 + dy**2 + dz**2)**0.5
            fij  = 48*(rij**-13) - 24*(rij**-7)
            pe  += rij**-12 - rij**-6
            # get unit vectors
            dxr = dx/rij
            dyr = dy/rij
            dzr = dz/rij
            # get pair force components
            fxn[i] += fij*dxr
            fyn[i] += fij*dyr
            fzn[i] += fij*dzr
            fxn[j] -= fij*dxr
            fyn[j] -= fij*dyr
            fzn[j] -= fij*dzr
    return fxn,fyn,fzn,4.0*pe
#----------------------------------------------

# function to calculate kinetic energy
#----------------------------------------------
cpdef kinetic(np.ndarray[double] vxn, np.ndarray[double] vyn, np.ndarray[double] vzn):
    # declare C-type variables
    cdef double ke
    # get kinetic energy
    ke  = 0.5*np.sum((vxn**2 + vyn**2 + vzn**2))
    return ke
#----------------------------------------------

# function to calculate FCC order parameter
#----------------------------------------------
cpdef orderFCC(np.ndarray[double] pxn, np.ndarray[double] pyn, np.ndarray[double] pzn,\
               double a, int nprt):
    # declare C-type variables
    cdef double ordX,ordY,ordZ,ordP
    # get order parameter
    ordX = np.sum(np.cos(4.0*np.pi*pxn/a))
    ordY = np.sum(np.cos(4.0*np.pi*pyn/a))
    ordZ = np.sum(np.cos(4.0*np.pi*pzn/a))
    ordP = (ordX + ordY + ordZ)/(3.0*nprt)
    return ordP
#----------------------------------------------