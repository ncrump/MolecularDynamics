"""
Created on Thu Jan 22 15:12:55 2015
CSI 786, Assignment 0
Nick Crump
"""

# Simple Harmonic Motion Animation
"""
Animation plot of position, velocity, energy and phase space
trajectory of 1D simple harmonic oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# define input parameters
c = 1     # amplitude
m = 1     # mass
phi = 0   # phase
omg = 1   # angular freq
tmx = 20  # max time

# setup subplot windows
fig = plt.figure()
fig.suptitle('1D Simple Harmonic Oscillator',fontsize=14)
fig.subplots_adjust(wspace=0.3)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(4,2,2)
ax3 = fig.add_subplot(4,2,4)
ax4 = fig.add_subplot(4,2,6)
ax5 = fig.add_subplot(4,2,8)
ax6 = fig.add_subplot(2,2,3)

# setup subplot axes
ax1.set_xlim(-c-1.0,c+0.25)
ax1.set_ylim(-0.5,0.5)
ax1.set_yticks([])
ax1.set_ylabel('Displacement',fontsize=14)
ax1.grid()

ax2.set_xlim(0,tmx)
ax2.set_ylim(-c,c)
ax2.set_ylabel('x',fontsize=14)
ax2.yaxis.set_label_position('right')
ax2.grid()

ax3.set_xlim(0,tmx)
ax3.set_ylim(-c,c)
ax3.set_ylabel('v',fontsize=14)
ax3.yaxis.set_label_position('right')
ax3.grid()

ax4.set_xlim(0,tmx)
ax4.set_ylim(0,0.5*m*c**2)
ax4.set_ylabel('PE',fontsize=14)
ax4.yaxis.set_label_position('right')
ax4.grid()

ax5.set_xlim(0,tmx)
ax5.set_ylim(0,0.5*m*c**2)
ax5.set_xlabel('t',fontsize=14)
ax5.set_ylabel('KE',fontsize=14)
ax5.yaxis.set_label_position('right')
ax5.grid()

ax6.set_xlim(-c,c)
ax6.set_ylim(-c,c)
ax6.set_xlabel('x',fontsize=14)
ax6.set_ylabel('v',fontsize=14)
ax6.grid()

# initialize plot data
line1, = ax1.plot([], [],'mo',markersize=24)
line2, = ax2.plot([], [],'b',lw=2)
line3, = ax3.plot([], [],'r',lw=2)
line4, = ax4.plot([], [],'b',lw=2)
line5, = ax5.plot([], [],'r',lw=2)
line6a, = ax6.plot([], [],'g',lw=2)
line6b, = ax6.plot([], [],'go',lw=2)
t,x,v,P,K = [],[],[],[],[]

# define data generator function
def gendata():
    # loop to generate plot data
    for t in np.arange(0,tmx,0.01):
        x = c*np.sin(omg*t + phi)
        v = c*omg*np.cos(omg*t + phi)
        P = 0.5*m*(c**2)*(omg**2)*(np.sin(omg*t + phi))**2
        K = 0.5*m*(c**2)*(omg**2)*(np.cos(omg*t + phi))**2
        yield t,x,v,P,K

# define update plot function
def genplot(data):
    # update plot data
    d1,d2,d3,d4,d5 = data
    t.append(d1)
    x.append(d2)
    v.append(d3)
    P.append(d4)
    K.append(d5)
    line1.set_data(x[-1],0)
    line2.set_data(t,x)
    line3.set_data(t,v)
    line4.set_data(t,P)
    line5.set_data(t,K)
    line6a.set_data(x,v)
    line6b.set_data(x[-1],v[-1])
    return line1,line2,line3,line4,line5,line6a,line6b,

ani = anim.FuncAnimation(fig,genplot,gendata,interval=10,blit=True,repeat=False)
plt.show()