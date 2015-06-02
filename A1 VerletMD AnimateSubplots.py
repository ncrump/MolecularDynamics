"""
Created on Fri Feb 06 21:30:57 2015
CSI 786, Assignment 1
Nick Crump
"""

# Simple MD Verlet-Velocity Method
"""
Molecular dynamics animation of Lennard-Jones particles
in 2D using Verlet-Velocity finite difference method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# function to calculate LJ force
def force(px,py,fx,fy,nprt):
    for i in range(nprt):
        fxi,fyi = 0,0
        for j in range(nprt):
            if i != j:
                dx = px[i]-px[j]
                dy = py[i]-py[j]
                rij = (dx**2 + dy**2)**0.5
                fij = 48*(rij**-13) - 24*(rij**-7)
                fxi += fij*(dx/rij)
                fyi += fij*(dy/rij)
        fx[i] = fxi
        fy[i] = fyi
    return fx,fy

# define input parameters
nprt = 2
nstp = 1000
dt   = 0.01
r0   = 1.2

# initialize positions
pxn = np.arange(0,nprt*r0,r0)
pyn = np.arange(0,nprt*r0,r0)
#pxn = np.array([0,1.2,2.4,3.6,0,1.2,2.4,3.6,0,1.2,2.4,3.6,0,1.2,2.4,3.6])
#pyn = np.array([0,0,0,0,1.2,1.2,1.2,1.2,2.4,2.4,2.4,2.4,3.6,3.6,3.6,3.6])

# initialize velocities
vxn = np.random.normal(0,0.2,nprt)
vyn = np.random.normal(0,0.2,nprt)

# initialize forces
fxi = np.zeros(nprt)
fyi = np.zeros(nprt)
fxn = np.zeros(nprt)
fyn = np.zeros(nprt)

# storage arrays
apx,apy,avx,avy = [pxn],[pyn],[vxn],[vyn]

# calculate initial forces
fxn,fyn = force(pxn,pyn,fxn,fyn,nprt)

# advance time step
step = 1
while step < nstp:
    # update pos
    pxn = pxn + vxn*dt + 0.5*(fxn*dt**2)
    pyn = pyn + vyn*dt + 0.5*(fyn*dt**2)

    # update force
    fxi = np.copy(fxn)
    fyi = np.copy(fyn)
    fxn,fyn = force(pxn,pyn,fxn,fyn,nprt)

    # update vel
    vxn = vxn + 0.5*(fxi+fxn)*dt
    vyn = vyn + 0.5*(fyi+fyn)*dt

    # store and increment step
    apx.append(pxn)
    apy.append(pyn)
    avx.append(vxn)
    avy.append(vyn)
    step += 1

# generate animation
# setup subplot windows
fig = plt.figure()
fig.suptitle('Molecular Dynamics of Lennard-Jones Particles',fontsize=14)
fig.subplots_adjust(wspace=0.3)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(4,2,2)
ax3 = fig.add_subplot(4,2,4)
ax4 = fig.add_subplot(4,2,6)
ax5 = fig.add_subplot(4,2,8)
ax6 = fig.add_subplot(2,2,3)

# setup subplot axes
lst = range(nstp)
px1min = min([apx[i][0] for i in lst])
py1min = min([apy[i][0] for i in lst])
vx1min = min([avx[i][0] for i in lst])
vy1min = min([avy[i][0] for i in lst])
px1max = max([apx[i][0] for i in lst])
py1max = max([apy[i][0] for i in lst])
vx1max = max([avx[i][0] for i in lst])
vy1max = max([avy[i][0] for i in lst])

ax1.set_xlim(np.min(apx)-1,np.max(apx)+1)
ax1.set_ylim(np.min(apy)-1,np.max(apy)+1)
ax1.set_ylabel('',fontsize=14)

ax2.set_xlim(0,nstp)
ax2.set_ylim(px1min,px1max)
ax2.set_ylabel('x$_{1}$',fontsize=14)
ax2.yaxis.set_label_position('right')

ax3.set_xlim(0,nstp)
ax3.set_ylim(py1min,py1max)
ax3.set_ylabel('y$_{1}$',fontsize=14)
ax3.yaxis.set_label_position('right')

ax4.set_xlim(0,nstp)
ax4.set_ylim(vx1min,vx1max)
ax4.set_ylabel('Vx$_{1}$',fontsize=14)
ax4.yaxis.set_label_position('right')

ax5.set_xlim(0,nstp)
ax5.set_ylim(vy1min,vy1max)
ax5.set_xlabel('Step',fontsize=14)
ax5.set_ylabel('Vy$_{1}$',fontsize=14)
ax5.yaxis.set_label_position('right')

ax6.set_xlim(px1min-0.2,px1max+0.2)
ax6.set_ylim(vx1min-0.2,vx1max+0.2)
ax6.set_xlabel('x$_{1}$',fontsize=14)
ax6.set_ylabel('Vx$_{1}$',fontsize=14)

# initialize plot data
line1, = ax1.plot([], [],'bo',markersize=20)
line2, = ax2.plot([], [],'g',lw=2)
line3, = ax3.plot([], [],'r',lw=2)
line4, = ax4.plot([], [],'g',lw=2)
line5, = ax5.plot([], [],'r',lw=2)
line6a, = ax6.plot([], [],'b',lw=2)
line6b, = ax6.plot([], [],'bo',lw=2)
n,px,py,px1,py1,vx1,vy1 = [],[],[],[],[],[],[]

# define data generator function
def gendata():
    # loop to generate plot data
    line1.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    line4.set_data([],[])
    line5.set_data([],[])
    line6a.set_data([],[])
    line6b.set_data([],[])
    return line1,line2,line3,line4,line5,line6a,line6b,

# define update plot function
def genplot(i):
    # update plot data
    n.append(i)
    px.append(apx[i])
    py.append(apy[i])
    px1.append(apx[i][0])
    py1.append(apy[i][0])
    vx1.append(avx[i][0])
    vy1.append(avy[i][0])
    line1.set_data(px[-1],py[-1])
    line2.set_data(n,px1)
    line3.set_data(n,py1)
    line4.set_data(n,vx1)
    line5.set_data(n,vy1)
    line6a.set_data(px1,vx1)
    line6b.set_data(px1[-1],vx1[-1])
    return line1,line2,line3,line4,line5,line6a,line6b,

ani = anim.FuncAnimation(fig,genplot,init_func=gendata,frames=nstp,interval=10,blit=True,repeat=False)
plt.show()