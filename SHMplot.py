"""
plot 1D simple harmonic motion
"""

import numpy as np
import matplotlib.pyplot as plt

# define input parameters
c = 1     # amplitude
m = 1     # mass
phi = 0   # phase
omg = 1   # angular freq
tmx = 30  # max time

# generate equations of motion
t = np.arange(0,tmx,0.01)
x = c*np.sin(omg*t + phi)
v = c*omg*np.cos(omg*t + phi)
P = 0.5*m*(c**2)*(omg**2)*(np.sin(omg*t + phi))**2
K = 0.5*m*(c**2)*(omg**2)*(np.cos(omg*t + phi))**2
E = K+P

# make subplots
# plot phsase space
plt.figure()
plt.subplot(2,2,1)
plt.plot(v,x)
plt.xlabel('x')
plt.ylabel('v')
# plot pos, vel
plt.subplot(4,2,2)
plt.plot(t,x,'b-')
plt.xticks([])
plt.ylabel('x')
plt.subplot(4,2,4)
plt.plot(t,v,'r-')
plt.xticks([])
plt.ylabel('v')
# plot energies
plt.subplot(4,2,6)
plt.plot(t,P,'b-')
plt.xticks([])
plt.ylabel('PE')
plt.subplot(4,2,8)
plt.plot(t,K,'r-')
plt.ylabel('KE')
plt.xlabel('t')
# plot total energy
plt.subplot(2,2,3)
plt.plot(t,E,'g-')
plt.ylabel('E')
plt.xlabel('t')