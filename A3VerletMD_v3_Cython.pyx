"""
Created on Sun Feb 15 20:47:16 2015
CSI 786, Assignment 3
Nick Crump
"""

# Version 3 Cython
# MD using Verlet-velocity method
# Periodic boundary conditions
# Minimum image convention
# Cut and shift of potential
# System relative to CM
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
    cdef int    nprt,nstp,cell,step
    cdef double dt,hlfdt,a,d,L,hlfL,Rc,Uc,Pi,Ki,Ei,Ti,order
    cdef double aveP,aveK,aveE,aveT,aveOldP,aveOldK,aveOldE,aveOldT
    cdef double tmpP,tmpK,tmpE,tmpT,stdP,stdK,stdE,stdT

    # declare C-type arrays
    cdef np.ndarray[double,ndim=1] pxn,pyn,pzn,pcm,vxn,vyn,vzn,vcm,fxn,fyn,fzn

    # define input parameters
    #----------------------------------------------
    nstp = 30000     # number of steps
    dt   = 0.005     # time step
    a    = 1.6       # side length of lattice
    cell = 1         # translations of unit cell
    #----------------------------------------------

    # get run parameters from input
    d     = 0.707*a           # distance between atoms
    L     = (cell+1)*a        # side length of box
    Rc    = 0.48*L            # LJ potential cutoff
    Uc    = Rc**-12 - Rc**-6  # LJ potential shift
    hlfdt = 0.5*dt            # half dt
    hlfL  = 0.5*L             # half L

    # print number of atoms in lattice
    nprt = 4*(cell+1)**3
    print '\natoms = ',nprt

    # initialize positions to FCC lattice
    pxn,pyn,pzn = lat.FCC(d,cell)

    # initialize velocities to zero
    vxn = np.zeros(nprt)
    vyn = np.zeros(nprt)
    vzn = np.zeros(nprt)

    # shift pos/vel wrt to CM
    pcm = np.zeros(3)
    vcm = np.zeros(3)
    pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm = CM(pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm,nprt)

    # initialize forces to zero
    fxn = np.zeros(nprt)
    fyn = np.zeros(nprt)
    fzn = np.zeros(nprt)

    # get initial values
    fxn,fyn,fzn,aveP     = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc)
    aveK                 = 0.5*np.sum((vxn*vxn + vyn*vyn + vzn*vzn))
    aveE                 = aveP + aveK
    aveT                 = (2*aveK)/(3*nprt-3)
    order                = orderFCC(pxn,pyn,pzn,pcm,a,nprt)
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
        pxn = pxn + vxn*dt + fxn*hlfdt*dt
        pyn = pyn + vyn*dt + fyn*hlfdt*dt
        pzn = pzn + vzn*dt + fzn*hlfdt*dt

        # apply periodic boundary conditions
        pxn = pxn - np.round(pxn/L)*L
        pyn = pyn - np.round(pyn/L)*L
        pzn = pzn - np.round(pzn/L)*L

        # update partial velocities at old step
        vxn = vxn + fxn*hlfdt
        vyn = vyn + fyn*hlfdt
        vzn = vzn + fzn*hlfdt

        # update forces
        fxn,fyn,fzn,Pi = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc)

        # update partial velocities at new step
        vxn = vxn + fxn*hlfdt
        vyn = vyn + fyn*hlfdt
        vzn = vzn + fzn*hlfdt

        # get energies and temp
        Ki = 0.5*np.sum((vxn*vxn + vyn*vyn + vzn*vzn))
        Ei = Pi + Ki
        Ti = (2*Ki)/(3*nprt-3)

        # get FCC order
        order = orderFCC(pxn,pyn,pzn,pcm,a,nprt)

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

    # check system CM at zero
    pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm = CM(pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm,nprt)

    # print averages
    print 'step = ',step
    print '\n<T> = %8.3f sigT = %7.6f' % (aveT,stdT)
    print '<P> = %8.3f sigP = %7.6f' % (aveP,stdP)
    print '<K> = %8.3f sigK = %7.6f' % (aveK,stdK)
    print '<E> = %8.3f sigE = %7.6f' % (aveE,stdE)
    print 'pCM = %8.3f %8.3f %8.3f' % (pcm[0],pcm[1],pcm[2])
    print 'vCM = %8.3f %8.3f %8.3f' % (vcm[0],vcm[1],vcm[2])
    print 'ord = %8.3f' % order

    # print runtime
    print 'runtime =',datetime.now()-t0

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
    plt.hist(Parr,bins=20)
    plt.ylabel('$PE$',fontsize=14)
    plt.subplot(2,1,2)
    plt.hist(Karr,bins=20)
    plt.ylabel('$KE$',fontsize=14)

    # plot FCC order parameter
    plt.figure()
    plt.plot(t,ordArr,'g-')
    plt.ylabel('FCC Order')
    plt.xlabel('Time')

# function to calculate LJ potential and forces
# applies minimum image convention for PBC
# applies cut and shift of potential
#----------------------------------------------
cpdef LJ(np.ndarray[double] pxn, np.ndarray[double] pyn, np.ndarray[double] pzn,\
         np.ndarray[double] fxn, np.ndarray[double] fyn, np.ndarray[double] fzn,\
         int nprt, double L, double hlfL, double Rc, double Uc):
    # declare C-type variables
    cdef int    i,j
    cdef double pe,dx,dy,dz,rij,fij,fx,fy,fz
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
            # apply minimum image convention
            if   dx >  hlfL: dx = dx - L
            elif dx < -hlfL: dx = dx + L
            if   dy >  hlfL: dy = dy - L
            elif dy < -hlfL: dy = dy + L
            if   dz >  hlfL: dz = dz - L
            elif dz < -hlfL: dz = dz + L
            rij  = (dx*dx + dy*dy + dz*dz)**0.5
            rij6 = rij*rij*rij*rij*rij*rij
            # get cut/shifted pair interactions
            if rij <= Rc:
                fij  = 48.0/(rij6*rij6*rij) - 24.0/(rij6*rij)
                pe   = pe + 1.0/(rij6*rij6) - 1.0/(rij6) - Uc
                # get component forces
                fx = fij*dx/rij
                fy = fij*dy/rij
                fz = fij*dz/rij
                # store forces
                fxn[i] = fxn[i] + fx
                fyn[i] = fyn[i] + fy
                fzn[i] = fzn[i] + fz
                fxn[j] = fxn[j] - fx
                fyn[j] = fyn[j] - fy
                fzn[j] = fzn[j] - fz
    return fxn,fyn,fzn,4.0*pe
#----------------------------------------------

# function to shift pos/vel wrt system CM
#----------------------------------------------
cpdef CM(np.ndarray[double] pxn, np.ndarray[double] pyn, np.ndarray[double] pzn,\
         np.ndarray[double] vxn, np.ndarray[double] vyn, np.ndarray[double] vzn,\
         np.ndarray[double] pcm, np.ndarray[double] vcm, int nprt):
    # get CM of pos/vel
    pcm[0],pcm[1],pcm[2] = np.sum(pxn), np.sum(pyn), np.sum(pzn)
    vcm[0],vcm[1],vcm[2] = np.sum(vxn), np.sum(vyn), np.sum(vzn)
    pcm,vcm = pcm/nprt, vcm/nprt
    # shift pos/vel wrt CM
    pxn,pyn,pzn = pxn-pcm[0], pyn-pcm[1], pzn-pcm[2]
    vxn,vyn,vzn = vxn-vcm[0], vyn-vcm[1], vzn-vcm[2]
    return pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm
#----------------------------------------------

# function to calculate FCC order parameter
#----------------------------------------------
cpdef orderFCC(np.ndarray[double] pxn, np.ndarray[double] pyn, np.ndarray[double] pzn,\
               np.ndarray[double] pcm, double a, int nprt):
    # declare C-type variables
    cdef double ordX,ordY,ordZ,ordP
    # get order parameter wrt CM
    ordX = np.sum(np.cos(4.0*np.pi*(pxn-pcm[0])/a))
    ordY = np.sum(np.cos(4.0*np.pi*(pyn-pcm[1])/a))
    ordZ = np.sum(np.cos(4.0*np.pi*(pzn-pcm[2])/a))
    ordP = (ordX + ordY + ordZ)/(3.0*nprt)
    return ordP
#----------------------------------------------