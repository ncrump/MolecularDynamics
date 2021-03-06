"""
Created on Fri Mar 27 18:41:32 2015
CSI 786, Assignment 6
Nick Crump
"""

# Version 5 Cython
# MD using Verlet-velocity method
# Periodic boundary conditions
# Minimum image convention
# Cut and shift of potential
# System relative to CM
# FCC order parameter
# Boltzmann H-function
# Radial distribution function
# Virial pressure
# Long range potential correction
# Long range pressure correction

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
    cdef int    eql,prd,neql,nstp,cell,nprt,step,hbins,gbins
    cdef double dt,hlfdt,a,d,L,hlfL,rho,Rc,Uc,Pi,Ki,Ei,Ti,Vi,temp,order,Hfun
    cdef double aveP,aveK,aveE,aveT,aveV,aveOldP,aveOldK,aveOldE,aveOldT,aveOldV
    cdef double tmpP,tmpK,tmpE,stdP,stdK,stdE,vmax,dv,dr,uLR,vLR

    # declare C-type arrays
    cdef np.ndarray[double,ndim=1] pxn,pyn,pzn,pcm,vxn,vyn,vzn,vcm,fxn,fyn,fzn
    cdef np.ndarray[double,ndim=1] vsqr,vmag,vhist,vbins,teql,tprd,grN,grR

    # define input parameters
    #----------------------------------------------
    eql   = 10000   # steps for equilibration
    prd   = 10000   # steps for production
    dt    = 0.005    # time step
    a     = 1.6      # side length of lattice
    cell  = 3        # translations of unit cell
    temp  = 2.0      # system temperature
    hbins = 20       # number of bins for H-function
    gbins = 120      # number of bins for g(r)
    #----------------------------------------------

    # get run parameters from input
    nprt  = 4*(cell+1)**3     # total number of atoms
    nstp  = eql+prd           # total number of steps
    d     = 0.707*a           # distance between atoms
    L     = (cell+1)*a        # side length of box
    hlfL  = 0.5*L             # half L
    hlfdt = 0.5*dt            # half dt
    rho   = nprt/(L*L*L)      # lattice number density
    Rc    = 0.48*L            # LJ potential cutoff
    Uc    = Rc**-12 - Rc**-6  # LJ potential shift

    # get long range correction terms
    uLR   = -(8*np.pi*rho)/(3*Rc**3)      # potential correction
    vLR   = -(16*np.pi*rho**2)/(3*Rc**3)  # virial correction

    # print number of atoms in lattice
    print '\natoms = ',nprt

    # initialize positions
    pxn,pyn,pzn = lat.FCC(d,cell)

    # initialize velocities
    vxn = np.random.normal(0,temp**0.5,nprt)
    vyn = np.random.normal(0,temp**0.5,nprt)
    vzn = np.random.normal(0,temp**0.5,nprt)

    # shift to CM
    pcm = np.zeros(3)
    vcm = np.zeros(3)
    pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm = CM(pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm,nprt)

    # initialize bins for g(r)
    dr  = Rc/gbins
    grN = np.zeros(gbins)
    grR = np.arange(1,gbins+1)*dr

    # initialize forces
    fxn = np.zeros(nprt)
    fyn = np.zeros(nprt)
    fzn = np.zeros(nprt)
    fxn,fyn,fzn,Pi,Vi,grN = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc,grN,dr)

    # initialize bins for H-function
    vsqr  = vxn*vxn + vyn*vyn + vzn*vzn
    vmag  = vsqr**0.5
    vmax  = np.max(vmag)
    dv    = vmax/hbins
    vhist = np.zeros(hbins)
    vbins = np.arange(1,hbins+1)*dv

    # initialize time arrays
    teql = np.arange(eql)*dt
    tprd = np.arange(prd)*dt

    # initialize storage arrays
    Parr,aveParr,stdParr = [],[],[]
    Karr,aveKarr,stdKarr = [],[],[]
    Earr,aveEarr,stdEarr = [],[],[]
    Tarr,aveTarr         = [],[]
    Varr,aveVarr         = [],[]
    ordArr,HfnArr        = [],[]

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
        fxn,fyn,fzn,Pi,Vi,grN = LJ(pxn,pyn,pzn,fxn,fyn,fzn,nprt,L,hlfL,Rc,Uc,grN,dr)

        # update partial velocities at new step
        vxn = vxn + fxn*hlfdt
        vyn = vyn + fyn*hlfdt
        vzn = vzn + fzn*hlfdt

        # get energies, temp, press
        vsqr = vxn*vxn + vyn*vyn + vzn*vzn
        Pi   = 4.0*Pi
        Ki   = 0.5*np.sum(vsqr)
        Ei   = Pi + Ki
        Ti   = (2*Ki)/(3*nprt-3)
        Vi   = rho*Ti - (rho*Vi)/(3*nprt-3)

        # get energies per atom
        Pi = Pi/nprt
        Ki = Ki/nprt
        Ei = Ei/nprt

        # flag to start equilibration
        if step <= eql:
            # get FCC order and H-function
            vmag = vsqr**0.5
            order = orderFCC(pxn,pyn,pzn,pcm,a,nprt)
            Hfun  = Hfunc(vmag,vbins,vhist,vmax,dv,hbins,nprt)
            # store values
            ordArr.append(order)
            HfnArr.append(Hfun)

        # flag to initialize averages
        if step == eql:
            # initial averages
            aveP = Pi
            aveK = Ki
            aveE = Ei
            aveT = Ti
            aveV = Vi
            # initial variances
            tmpP,tmpK,tmpE  = 0,0,0
            # zero g(r) histogram
            grN = grN*0

        # flag to start production
        if step >= eql:
            # update running averages
            aveOldP = aveP
            aveOldK = aveK
            aveOldE = aveE
            aveOldT = aveT
            aveOldV = aveV
            aveP = aveP + (Pi-aveP)/step
            aveK = aveK + (Ki-aveK)/step
            aveE = aveE + (Ei-aveE)/step
            aveT = aveT + (Ti-aveT)/step
            aveV = aveV + (Vi-aveV)/step
            tmpP = tmpP + (Pi-aveOldP)*(Pi-aveP)
            tmpK = tmpK + (Ki-aveOldK)*(Ki-aveK)
            tmpE = tmpE + (Ei-aveOldE)*(Ei-aveE)
            stdP = (tmpP/step)**0.5
            stdK = (tmpK/step)**0.5
            stdE = (tmpE/step)**0.5
            # store values
            Parr.append(Pi)
            Karr.append(Ki)
            Earr.append(Ei)
            Tarr.append(Ti)
            Varr.append(Vi)
            aveParr.append(aveP)
            aveKarr.append(aveK)
            aveEarr.append(aveE)
            aveTarr.append(aveT)
            aveVarr.append(aveV)
            stdParr.append(stdP)
            stdKarr.append(stdK)
            stdEarr.append(stdE)

        # increment step
        step += 1

    # normalize g(r)
    grN = grN/(4*np.pi*rho*prd*nprt*dr*grR*grR)

    # check CM at zero
    pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm = CM(pxn,pyn,pzn,vxn,vyn,vzn,pcm,vcm,nprt)

    # print averages
    print 'step = ',step
    print '\n<P> = %8.3f sigP = %7.6f' % (aveP,stdP)
    print '<K> = %8.3f sigK = %7.6f' % (aveK,stdK)
    print '<E> = %8.3f sigE = %7.6f' % (aveE,stdE)
    print '<T> = %8.3f ' % aveT
    print '<V> = %8.3f ' % aveV
    print 'PLR = %8.3f ' % uLR
    print 'VLR = %8.3f ' % vLR
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

    # plot temperature and pressure
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(tprd,Tarr,'b-')
    plt.plot(tprd,aveTarr,'r-')
    plt.ylabel('$<T>$',fontsize=14)
    plt.subplot(2,1,2)
    plt.plot(tprd,Varr,'b-')
    plt.plot(tprd,aveVarr,'r-')
    plt.ylabel('$<P>$',fontsize=14)
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
    plt.ylim(0,1)
    plt.ylabel('FCC Order')
    plt.xlabel('Time')

    # plot Boltzmann H-function
    plt.figure()
    plt.plot(teql,HfnArr,'g-')
    plt.ylabel('H - Function')
    plt.xlabel('Time')

    # plot g(r)
    plt.figure()
    plt.plot(grR,grN,'r-')
    plt.ylabel('$g\ (r)$',fontsize=14)
    plt.xlabel('$r$',fontsize=14)

# function to calculate LJ potential, LJ forces, g(r), virial pressure
# applies minimum image convention for PBC
# applies cut and shift of potential
#----------------------------------------------
cpdef LJ(np.ndarray[double] pxn, np.ndarray[double] pyn, np.ndarray[double] pzn,\
         np.ndarray[double] fxn, np.ndarray[double] fyn, np.ndarray[double] fzn,\
         int nprt, double L, double hlfL, double Rc, double Uc,\
         np.ndarray[double] grN, double dr):
    # declare C-type variables
    cdef int    i,j,indx
    cdef double pe,vp,dx,dy,dz,rij,fij,fx,fy,fz
    # zero forces and potential
    fxn,fyn,fzn = fxn*0,fyn*0,fzn*0
    pe,vp = 0,0
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
                # get g(r) histogram
                indx = int(rij/dr)
                grN[indx] += 2
                # get virial coefficient
                vp = vp + fij*rij
    return fxn,fyn,fzn,pe,vp,grN
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
            double vmax, double dv, int hbins, int nprt):
    # declare C-type variables
    cdef int i,indx
    cdef double hfunc
    # get velocity histogram
    for i in range(nprt):
        indx = int(vmag[i]/dv)-1
        if vmag[i] < vmax: vhist[indx] += 1
    # normalize velocity histogram
    vhist = vhist/np.sum(vhist)
    # get H-function
    hfunc = 0
    for i in range(hbins):
        if vhist[i] > 0: hfunc = hfunc + vhist[i]*np.log(vhist[i])*vbins[i]
    return hfunc
#----------------------------------------------