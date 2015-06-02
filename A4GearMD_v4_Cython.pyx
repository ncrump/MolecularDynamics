"""
Created on Sun Mar 08 17:23:22 2015
CSI 786, Assignment 5
Nick Crump
"""

# Version 5 Cython
# MD using Gear predictor-corrector method
# Periodic boundary conditions
# Minimum image convention
# Cut and shift of potential
# System relative to CM
# FCC order parameter
# Boltzmann H-function

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
    cdef int    wrm,eql,prd,neql,nstp,cell,nprt,step,bins
    cdef double dt,a,d,L,hlfL,Rc,Uc,Pi,Ki,Ei,Ti,order,Hfun
    cdef double dt2,dt3,dt4,dt5,q0,q1,q2,q3,q4,q5
    cdef double aveP,aveK,aveE,aveT,aveOldP,aveOldK,aveOldE,aveOldT
    cdef double tmpP,tmpK,tmpE,tmpT,stdP,stdK,stdE,stdT,vmax,dv

    # declare C-type arrays
    cdef np.ndarray[double,ndim=1] pxn,pyn,pzn,pcm,vxn,vyn,vzn,vcm,fxn,fyn,fzn
    cdef np.ndarray[double,ndim=1] px2,py2,pz2,px3,py3,pz3,px4,py4,pz4,px5,py5,pz5
    cdef np.ndarray[double,ndim=1] vsqr,vmag,vhist,vbins,teql,tprd,dfx,dfy,dfz

    # define input parameters
    #----------------------------------------------
    wrm  = 500      # steps for warmup
    eql  = 30000    # steps for equilibration
    prd  = 30000   # steps for production
    dt   = 0.005    # time step
    a    = 1.6      # side length of lattice
    cell = 2        # translations of unit cell
    bins = 10       # number of bins for H-function
    #----------------------------------------------

    # get run parameters from input
    neql  = wrm+eql           # steps after equilibration
    nstp  = wrm+eql+prd       # total number of steps
    d     = 0.707*a           # distance between atoms
    L     = (cell+1)*a        # side length of box
    Rc    = 0.48*L            # LJ potential cutoff
    Uc    = Rc**-12 - Rc**-6  # LJ potential shift
    hlfL  = 0.5*L             # half L

    # get Gear parameters
    dt2 = dt**2/2.0
    dt3 = dt**3/6.0
    dt4 = dt**4/24.0
    dt5 = dt**5/120.0
    q0  = 3/16.0
    q1  = 251/360.0
    q2  = 1.0
    q3  = 11/18.0
    q4  = 1/6.0
    q5  = 1/60.0

    # print number of atoms in lattice
    nprt = 4*(cell+1)**3
    print '\natoms = ',nprt

    # initialize positions
    pxn,pyn,pzn = lat.FCC(d,cell)

    # initialize velocities
    vxn,vyn,vzn = np.zeros(nprt),np.zeros(nprt),np.zeros(nprt)

    # shift to CM
    pcm,vcm = np.zeros(3),np.zeros(3)
    pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm = CM(pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm,nprt)

    # initialize forces
    fxn,fyn,fzn = np.zeros(nprt),np.zeros(nprt),np.zeros(nprt)
    fxn,fyn,fzn,Pi = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc)

    # initialize Gear higher order derivatives
    px2,py2,pz2 = np.zeros(nprt),np.zeros(nprt),np.zeros(nprt)
    px3,py3,pz3 = np.zeros(nprt),np.zeros(nprt),np.zeros(nprt)
    px4,py4,pz4 = np.zeros(nprt),np.zeros(nprt),np.zeros(nprt)
    px5,py5,pz5 = np.zeros(nprt),np.zeros(nprt),np.zeros(nprt)

    # initialize time arrays
    teql = np.arange(eql+prd)*dt
    tprd = np.arange(prd)*dt

    # initialize storage arrays
    Parr,aveParr,stdParr = [],[],[]
    Karr,aveKarr,stdKarr = [],[],[]
    Earr,aveEarr,stdEarr = [],[],[]
    Tarr,aveTarr,stdTarr = [],[],[]
    ordArr,HfnArr        = [],[]

    # advance time step: Gear predictor-corrector method
    step = 1
    while step < nstp:
        # predict positions from Taylor series
        pxn = pxn + vxn*dt + px2*dt2 + px3*dt3 + px4*dt4 + px5*dt5
        pyn = pyn + vyn*dt + py2*dt2 + py3*dt3 + py4*dt4 + py5*dt5
        pzn = pzn + vzn*dt + pz2*dt2 + pz3*dt3 + pz4*dt4 + pz5*dt5
        # predict velocities from Taylor series
        vxn = vxn + px2*dt + px3*dt2 + px4*dt3 + px5*dt4
        vyn = vyn + py2*dt + py3*dt2 + py4*dt3 + py5*dt4
        vzn = vzn + pz2*dt + pz3*dt2 + pz4*dt3 + pz5*dt4
        # predict derivatives from Taylor series
        px2 = px2 + px3*dt + px4*dt2 + px5*dt3
        py2 = py2 + py3*dt + py4*dt2 + py5*dt3
        pz2 = pz2 + pz3*dt + pz4*dt2 + pz5*dt3
        px3 = px3 + px4*dt + px5*dt2
        py3 = py3 + py4*dt + py5*dt2
        pz3 = pz3 + pz4*dt + pz5*dt2
        px4 = px4 + px5*dt
        py4 = py4 + py5*dt
        pz4 = pz4 + pz5*dt

        # update forces and get correcction terms
        fxn,fyn,fzn,Pi = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc)
        dfx,dfy,dfz    = (fxn-px2)*dt2,(fyn-py2)*dt2,(fzn-pz2)*dt2

        # correct positions
        pxn = pxn + q0*dfx
        pyn = pyn + q0*dfy
        pzn = pzn + q0*dfz
        # correct velocities
        vxn = vxn + q1*dfx/dt
        vyn = vyn + q1*dfy/dt
        vzn = vzn + q1*dfz/dt
        # correct derivatives
        px2 = px2 + q2*dfx/dt2
        py2 = py2 + q2*dfy/dt2
        pz2 = pz2 + q2*dfz/dt2
        px3 = px3 + q3*dfx/dt3
        py3 = py3 + q3*dfy/dt3
        pz3 = pz3 + q3*dfz/dt3
        px4 = px4 + q4*dfx/dt4
        py4 = py4 + q4*dfy/dt4
        pz4 = pz4 + q4*dfz/dt4
        px5 = px5 + q5*dfx/dt5
        py5 = py5 + q5*dfy/dt5
        pz5 = pz5 + q5*dfz/dt5

        # apply periodic boundary conditions
        pxn = pxn - np.round(pxn/L)*L
        pyn = pyn - np.round(pyn/L)*L
        pzn = pzn - np.round(pzn/L)*L

        # get energies and temp
        vsqr = vxn*vxn + vyn*vyn + vzn*vzn
        Ki   = 0.5*np.sum(vsqr)
        Ei   = Pi + Ki
        Ti   = (2*Ki)/(3*nprt-3)

        # get energies per atom
        Pi = Pi/nprt
        Ki = Ki/nprt
        Ei = Ei/nprt

        # flag to initialize H-function
        if step == wrm:
            # initialize bins for H-function
            vmag  = vsqr**0.5
            vmax  = np.max(vmag)
            dv    = vmax/bins
            vbins = np.arange(1,bins+1)*dv
            vhist = np.zeros(bins)

        # flag to start equilibration
        if step >= wrm:
            # get FCC order and H-function
            vmag = vsqr**0.5
            order = orderFCC(pxn,pyn,pzn,pcm,a,nprt)
            Hfun  = Hfunc(vmag,vbins,vhist,vmax,dv,bins,nprt)
            # store values
            ordArr.append(order)
            HfnArr.append(Hfun)

        # flag to initialize averages
        if step == neql:
            # initial averages
            aveP = Pi
            aveK = Ki
            aveE = Ei
            aveT = Ti
            # initial variances
            tmpP,tmpK,tmpE,tmpT  = 0,0,0,0

        # flag to start production
        if step >= neql:
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
            # store values
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

        # increment step
        step += 1

    # check CM at zero
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
    print 'H-f = %8.3f' % Hfun

    # print runtime
    print 'runtime =',datetime.now()-t0

    # plot potential energy
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tprd,Parr,'b-')
    plt.plot(tprd,aveParr,'r-')
    plt.ylabel('$<PE>$',fontsize=14)
    plt.subplot(2,1,2)
    plt.plot(tprd,stdParr,'r-')
    plt.ylabel(r'$\sigma _{PE}$',fontsize=16)
    plt.xlabel('Time')

    # plot kinetic energy
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tprd,Karr,'b-')
    plt.plot(tprd,aveKarr,'r-')
    plt.ylabel('$<KE>$',fontsize=14)
    plt.subplot(2,1,2)
    plt.plot(tprd,stdKarr,'r-')
    plt.ylabel(r'$\sigma _{KE}$',fontsize=16)
    plt.xlabel('Time')

    # plot total energy
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tprd,Earr,'b-')
    plt.plot(tprd,aveEarr,'r-')
    plt.ylabel('$<E>$',fontsize=14)
    plt.ticklabel_format(style='sci',useOffset=False,axis='y')
    plt.subplot(2,1,2)
    plt.plot(tprd,stdEarr,'r-')
    plt.ylabel(r'$\sigma _{E}$',fontsize=16)
    plt.xlabel('Time')

    # plot temperature
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tprd,Tarr,'b-')
    plt.plot(tprd,aveTarr,'r-')
    plt.ylabel('$<T>$',fontsize=14)
    plt.subplot(2,1,2)
    plt.plot(tprd,stdTarr,'r-')
    plt.ylabel(r'$\sigma _{T}$',fontsize=16)
    plt.xlabel('Time')

    # plot normalized histogram of energies
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(Parr,bins=20,normed=True)
    plt.ylabel('$PE$',fontsize=14)
    plt.subplot(2,1,2)
    plt.hist(Karr,bins=20,normed=True)
    plt.ylabel('$KE$',fontsize=14)

    # plot FCC order parameter
    plt.figure()
    plt.plot(teql,ordArr,'g-')
    plt.ylabel('FCC Order')
    plt.xlabel('Time')

    # plot Boltzmann H-function
    plt.figure()
    plt.plot(range(len(HfnArr)),HfnArr,'g-')
    plt.xlim(-5,teql[-1])
    plt.ylabel('H - Function')
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

# function to calculate Boltzmann H-function
#----------------------------------------------
cpdef Hfunc(np.ndarray[double] vmag, np.ndarray[double] vbins, np.ndarray[double] vhist,\
            double vmax, double dv, int bins, int nprt):
    # declare C-type variables
    cdef int i
    cdef double hfunc
    # get velocity histogram
    for i in range(nprt):
        if vmag[i] < vmax: vhist[int(vmag[i]/dv)-1] += 1
    # normalize velocity histogram
    vhist = vhist/np.sum(vhist)
    # get H-function
    hfunc = 0
    for i in range(bins):
        if vhist[i] > 0: hfunc = hfunc + vhist[i]*np.log(vhist[i])*vbins[i]
    return hfunc
#----------------------------------------------