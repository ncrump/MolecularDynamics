"""
Created on Sun Feb 15 20:47:16 2015
CSI 786, Assignment 3
Nick Crump
"""

# Version 3Test Cython
# Test for energy conservation
# MD using Verlet-velocity method
# Periodic boundary conditions
# Minimum image convention
# Cut and shift of potential
# System relative to CM

"""
Molecular dynamics simulation of Lennard-Jones atoms
Test for energy conservation
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
    cdef int    nprt,cell,step
    cdef double dt,hlfdt,a,d,L,hlfL,Rc,Uc,Pi,Ki,Ei,Ti,time,ttot
    cdef double aveP,aveK,aveE,aveT,aveOldP,aveOldK,aveOldE,aveOldT
    cdef double tmpP,tmpK,tmpE,tmpT,stdP,stdK,stdE,stdT

    # declare C-type arrays
    cdef np.ndarray[double,ndim=1] pxi,pyi,pzi,vxi,vyi,vzi
    cdef np.ndarray[double,ndim=1] pxn,pyn,pzn,pcm,vxn,vyn,vzn,vcm,fxn,fyn,fzn

    # define input parameters
    #----------------------------------------------
    ttot = 5         # total time
    a    = 1.6       # side length of lattice
    cell = 1         # translations of unit cell
    #----------------------------------------------

    # get run parameters from input
    d     = 0.707*a           # distance between atoms
    L     = (cell+1)*a        # side length of box
    Rc    = 0.48*L            # LJ potential cutoff
    Uc    = Rc**-12 - Rc**-6  # LJ potential shift
    hlfL  = 0.5*L             # half L

    # print number of atoms in lattice
    nprt = 4*(cell+1)**3
    print '\natoms = ',nprt

    # initialize positions to FCC lattice
    pxi,pyi,pzi = lat.FCC(d,cell)

    # initialize velocities to zero
    vxi = np.zeros(nprt)
    vyi = np.zeros(nprt)
    vzi = np.zeros(nprt)

    # shift pos/vel wrt to CM
    pcm = np.zeros(3)
    vcm = np.zeros(3)
    pxi,pyi,pzi,vxi,vyi,vzi,pcm,vcm = CM(pxi,pyi,pzi,vxi,vyi,vzi,pcm,vcm,nprt)

    # storage arrays
    dtarr = []
    sdarr = []

    # loop through dt
    for k in [0.05/(2**i) for i in range(8)]:

        dt    = k             # time step
        hlfdt = 0.5*dt        # half dt
        nstp  = int(ttot/dt)  # number of steps

        # reset positions and velocities
        pxn,pyn,pzn = pxi,pyi,pzi
        vxn,vyn,vzn = vxi,vyi,vzi

        # initialize forces to zero
        fxn = np.zeros(nprt)
        fyn = np.zeros(nprt)
        fzn = np.zeros(nprt)

        # get initial values
        fxn,fyn,fzn,aveP     = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc)
        aveK                 = 0.5*np.sum((vxn*vxn + vyn*vyn + vzn*vzn))
        aveE                 = aveP + aveK
        aveT                 = (2*aveK)/(3*nprt-3)
        tmpP,tmpK,tmpE,tmpT  = 0,0,0,0

        # advance time step: Verlet-velocity method
        step = 1
        time = dt
        while time < ttot:

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

            # increment step
            step += 1
            time += dt

        # store values
        dtarr.append(dt)
        sdarr.append(stdE**0.5)

        # print averages
        print 'dt = %1.3f <P> = %1.6f <K> = %1.6f <E> = %1.6f stdE = %1.6f' \
               % (dt,aveP,aveK,aveE,stdE)

    # print runtime
    print 'runtime =',datetime.now()-t0

    # make plot
    plt.figure()
    plt.plot(dtarr,sdarr,'bo-')
    plt.xlabel('$dt$',fontsize=16)
    plt.ylabel(r'$\sqrt{\sigma _{E}}$',fontsize=16)


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