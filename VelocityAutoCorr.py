"""
Created on Mon Apr 27 18:53:09 2015
CSI 786, Assignment 8
Nick Crump
"""

# Calculates velocity autocorrelation function
# from input data file

import numpy as np
import matplotlib.pyplot as plt

# set input parameters
# ------------------------------------
file1 = 'velocity_rho0.7_temp1.5.txt'
nprt  = 256    # number of atoms
tmax =  1.6    # max time delay
# ------------------------------------
# note: max time delay is limit for periodic systems
# tmax based on periodic correlation time for nprt

# load input file
t,v = np.loadtxt(file1,unpack=True)

# get index and time delays
nmax = int(tmax/t[nprt])
nval = len(t)
delt = np.arange(1,nmax)

# loop over time delays
vacf = []
for k in delt:
    nstp = 0
    vsum = 0
    # loop over time origins for all atoms
    for j in range(0,nval-nprt*k,nprt):
        nstp += 1
        vsum = vsum + np.sum(v[j:j+nprt]*v[j+nprt*k:j+nprt*k+nprt])
    vacf.append(vsum/(nstp*nprt))

# normalize function and make time array
vacf = np.array(vacf)/vacf[0]
t = delt*t[nprt]

# plot
plt.figure()
plt.plot(t,vacf,'g',label='T=1.5')
plt.ylabel('VACF')
plt.xlabel('Delay Time')
plt.legend(loc=1,fontsize=13)